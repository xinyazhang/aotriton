// Copyright © 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// perfmon-exec0.md T07: the exaid line protocol, core-side.
//
// Implements exactly the wire format `python/tune/exaid.py`'s `ExaidProxy`
// already speaks (`ExaidProxy.write`/`ExaidProxy.readinfo`,
// python/tune/exaid.py:58-95):
//
//   * Commands arrive newline-delimited on stdin, whitespace-separated
//     tokens (`ExaidProxy.write` joins `*objects` with a single space and
//     appends '\n'; `exaid.py`'s own `first(line, sep=" ")` splits a
//     RESPONSE the same way -- this module applies the identical splitting
//     rule to incoming COMMAND lines, since exec0.md T07 specifies commands
//     as space-separated tokens: `measure <entry_pon> <iface> <backend>`,
//     `enumerate <entry_pon> <iface>`, `platform`, `exit`).
//   * Success -> emit `OK <json>\n`, flushed.
//   * Failure -> emit a single line that does NOT start with "OK", flushed;
//     `ExaidProxy.readinfo` turns any such line into `ExaidSubprocessNotOK`.
//   * Progress/waiting -> emit `OVERHEATING: <detail>\n`, flushed;
//     `ExaidProxy.readinfo` loops past these without resetting its timeout
//     budget or otherwise treating them as an error (see `readinfo`'s
//     `if ret == "OVERHEATING:": continue`). Note the trailing colon is
//     part of the first whitespace-delimited token itself -- the literal
//     text emitted must be "OVERHEATING:" immediately followed by a space,
//     matching exaid.py's `first()` split exactly.
//
// This header purposefully has NO dependency on perfmon_abi.h, HIP, or
// AOTriton -- protocol.cc is pure C++ standard library, CPU-testable
// without a GPU (perfmon-exec0.md T07's whole point).

#ifndef PERFMON_CORE_PROTOCOL_H
#define PERFMON_CORE_PROTOCOL_H

#include <functional>
#include <istream>
#include <ostream>
#include <string>
#include <vector>

namespace perfmon {

// One decoded command line: `name` is the first whitespace-delimited
// token, `args` is every token after it. Extra internal whitespace between
// tokens is collapsed (matching `str.split()` semantics, not `str.split('
// ', maxsplit=1)` -- unlike exaid.py's own `first()`, which only splits
// once because its second field is a whole JSON blob; perfmon's incoming
// commands instead have a small, fixed number of scalar-token arguments
// per exec0.md T07's four command shapes, none of which is free-form JSON).
struct Command {
  std::string name;
  std::vector<std::string> args;
};

// Split `line` (already stripped of its trailing newline) into a Command
// on runs of whitespace. An empty or all-whitespace line yields an empty
// `name`.
Command parse_command(const std::string& line);

// Low-level line emitters. Each writes exactly one '\n'-terminated line to
// `out` and flushes it immediately -- ExaidProxy.readinfo blocks on
// readline, so a buffered-but-unflushed reply would hang the caller.
//
// `emit_ok`'s `json_body` is written verbatim after "OK "; callers are
// responsible for producing valid JSON (protocol.cc does not parse or
// validate it -- that mirrors exaid.py's own worker side, `testrun.py`,
// which is out of scope for perfmon's core).
void emit_ok(std::ostream& out, const std::string& json_body);

// `detail` becomes the entire failure line. It must not begin with "OK "
// (that would be misread as success) or "OVERHEATING:" (misread as a
// progress line) -- emit_error asserts this in debug builds rather than
// silently emitting an ambiguous line.
void emit_error(std::ostream& out, const std::string& detail);

// Emits "OVERHEATING: <detail>\n". Used by the thermal gate (T08) while it
// waits for the GPU to cool -- one line per poll, so a long wait is visible
// on the wire without ExaidProxy.readinfo ever timing out or erroring.
void emit_overheating(std::ostream& out, const std::string& detail);

// Handler table a runner main() supplies. Each handler receives the
// command's argument tokens and either returns a JSON body (to be wrapped
// in "OK <json>") or throws `std::runtime_error`, whose `what()` becomes
// the failure line via emit_error.
struct Handlers {
  std::function<std::string(const std::vector<std::string>&)> measure;
  std::function<std::string(const std::vector<std::string>&)> enumerate_cmd;
  std::function<std::string(const std::vector<std::string>&)> platform;
};

// Reads newline-delimited commands from `in` until an "exit" command or
// EOF, dispatching each to the matching entry of `handlers` and framing
// the result (or exception) per the OK/error rules above. An unrecognized
// command name is treated as a failure (emit_error), not a crash. Returns
// 0 on a clean "exit"/EOF, matching a process exit status the proxy's
// `join()` expects.
int protocol_loop(std::istream& in, std::ostream& out, const Handlers& handlers);

}  // namespace perfmon

#endif  // PERFMON_CORE_PROTOCOL_H
