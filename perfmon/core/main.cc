// Copyright © 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// perfmon-exec0.md T13 gap-fill: the runner executable's main().
//
// WHY THIS FILE EXISTS: T12/T13 both require a working
// `perfmon/subjects/<id>/bin/runner` -- T13's own Verify step even pipes
// "exit" into it and checks the exit code -- but no earlier task's Files
// list (`perfmon-exec0.md`, checked in full: T06 through T13's own "Files"
// rows) ever names a main.cc/runner.cc, and `protocol.h`'s own doc comment
// (T07) explicitly describes `Handlers`/`protocol_loop` as "a runner
// main() supplies" without that main() existing anywhere. Confirmed via
// `grep -rln "int main(" perfmon/` (only `probe_graph_timing.cc`, T05's
// documented throwaway hardware probe) before writing this. This is a
// genuine planning gap, not a per-task oversight -- written now, under T13
// (the task that actually needs a working runner binary), with every
// judgment call disclosed here and in entry_codec.h/.cc.
//
// Verified on gfx942 / ROCm 7.14 by T14: `enumerate`, `measure` (attn_fwd
// backends 0-1, attn_bwd backends 0-2), `platform` and `exit` all drive
// correctly over stdin, and an out-of-range backend index produces a
// non-`OK` line without crashing the process.
//
// Still unexercised here: the GQA `N_HEADS=(H, G)` wire form that rejoin()
// below exists to handle -- every T14 shape used a scalar `N_HEADS`.
//
// Architecture: statically links exactly one family's vtable via
// `pmon_family_entry()` (perfmon_abi.h) -- no dlopen (rev0 D4) -- and is
// therefore built PER SUBJECT, alongside `adapter_v3.cc`
// (modules/flash/perfmon/runner/CMakeLists.txt, amended by this same T13
// commit to add this file + entry_codec.cc as an `add_executable(runner
// ...)` target linked against the `perfmon_flash` target it already
// builds). The source lives under perfmon/core/ because its OWN content is
// family- and AOTriton-neutral -- it never names `flash`, `attn_fwd`, or
// any AOTriton type, only perfmon_abi.h's plain-C vtable -- even though the
// *executable* it becomes part of is per-subject. Nothing here prevents a
// second family (a future, non-flash family) from reusing this exact file
// unchanged.
//
// --- Disclosed design decisions (see entry_codec.h for the PON-parsing
// ones; these are the runner-level ones) ---
//
// * `argv[1]`, if present, is treated as this runner's `subject_id` (used
//   only to derive `pmon_entry.seed`, entry_codec.h item 3). T13's own
//   Verify step invokes the runner with NO argv at all
//   (`perfmon/subjects/<id>/bin/runner <<< "exit"`), so this cannot be a
//   required positional argument; it defaults to the empty string,
//   documented here as a placeholder until a real invocation convention is
//   decided (that decision belongs to a `runner_proxy.py`, out of this
//   task's T01-T13 scope).
// * A single non-default `hipStream_t`, created once at startup and reused
//   for every `measure`, since T12's spec requires `launch()` to run on a
//   capturable (non-null) stream and creating one per call would be wasted
//   work for a long-lived process.
// * JSON response shapes for `measure`/`enumerate`/`platform` are this
//   file's own first-cut choice -- no earlier task specifies them, and the
//   later task that would consume them (`python/tune/perfmon/runner_proxy.py`)
//   is out of scope here. Kept simple and hand-serialized (this project
//   has no JSON library dependency anywhere); a real consumer may require
//   a different shape, which is exactly the kind of thing T14's manual
//   smoke test (out of scope for this agent) would surface.
// * No live PMON_ABI_VERSION cross-check against the family library: doing
//   that properly needs a NEW exported symbol reporting the version the
//   family lib was compiled against (perfmon_abi.h's own vtable has no
//   such field), which would be a second ABI amendment in the same spirit
//   as T06's `PMON_ABI_VERSION 2` bump. Deliberately not done here to avoid
//   further, unbounded scope growth in a task whose Files list is
//   `perfmon/build_subject.sh`; flagged in perfmon-handoff0.md as a known
//   gap instead.

#include "entry_codec.h"
#include "perfmon_abi.h"
#include "protocol.h"
#include "timing.h"

#include <cstdio>
#include <exception>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

// Reassembles a possibly-fragmented `<entry_pon>` from `tokens[0..count)`
// (`count` tokens, single space between them) -- see entry_codec.h item 2
// for why a single PON token can arrive split into more than one
// whitespace-delimited wire token (a GQA `N_HEADS=(10, 2)` field's
// `repr()`-inserted ", "). Undoing that split by rejoining with exactly one
// space is exact: that comma-space is the ONLY source of extra whitespace
// `render_pon` ever produces for this family's field set.
std::string rejoin(const std::vector<std::string>& tokens, size_t count) {
  std::string out;
  for (size_t i = 0; i < count; ++i) {
    if (i) out += ' ';
    out += tokens[i];
  }
  return out;
}

std::string json_escape(const std::string& s) {
  std::string out;
  out.reserve(s.size() + 2);
  for (char c : s) {
    if (c == '"' || c == '\\') out += '\\';
    out += c;
  }
  return out;
}

// %.17g round-trips any double bit-for-bit back through strtod -- this
// project has no JSON library, so every number here is hand-serialized;
// round-tripping exactly is more important than a short/pretty rendering.
std::string json_double(double v) {
  char buf[64];
  std::snprintf(buf, sizeof(buf), "%.17g", v);
  return buf;
}

std::string json_thermal(const perfmon::pmon_thermal& t) {
  // T08's spec, wired through T10's MeasurementRecord: an ungated
  // measurement must serialize as `"thermal": null`, never a struct of
  // zeros that could be mistaken for a real zero reading.
  if (!t.valid) return "null";
  std::ostringstream oss;
  // `throttled` is null, not false, when the platform does not report
  // throttle state -- see pmon_thermal::throttle_known (thermal.h). gfx942
  // is such a platform, so this is the common case, not a corner one.
  oss << "{\"temp_c\":" << json_double(t.temp_c)
      << ",\"sclk_mhz\":" << json_double(t.sclk_mhz)
      << ",\"throttled\":"
      << (t.throttle_known ? (t.throttled ? "true" : "false") : "null") << "}";
  return oss.str();
}

struct Runner {
  const pmon_family_vtable* vtable = nullptr;
  hipStream_t stream = nullptr;
  std::string subject_id;

  std::string measure(const std::vector<std::string>& args) {
    // Wire shape (protocol.h, T07): `measure <entry_pon> <iface> <backend>`.
    // entry_pon may span more than one token -- see rejoin()'s doc comment.
    if (args.size() < 3) {
      throw std::runtime_error(
          "measure: expected at least 3 args (entry_pon iface backend), got " +
          std::to_string(args.size()));
    }
    const std::string entry_pon = rejoin(args, args.size() - 2);
    const int32_t iface = std::stoi(args[args.size() - 2]);
    const int32_t backend = std::stoi(args[args.size() - 1]);

    perfmon::PmonEntryFields ff = perfmon::parse_entry_pon(entry_pon, iface, subject_id);
    pmon_entry entry{};
    entry.iface = ff.iface;
    entry.dtype = ff.dtype;
    entry.hdim = ff.hdim;
    entry.seqlen_q = ff.seqlen_q;
    entry.seqlen_k = ff.seqlen_k;
    entry.causal = ff.causal;
    entry.dropout_p = ff.dropout_p;
    entry.bias_type = ff.bias_type;
    entry.batch = ff.batch;
    entry.n_heads = ff.n_heads;
    entry.gqa_ratio = ff.gqa_ratio;
    entry.varlen = ff.varlen;
    entry.storage_flip = ff.storage_flip;
    entry.seed = ff.seed;

    void* ctx = nullptr;
    int rc = vtable->prepare(&entry, backend, &ctx);
    if (rc != 0) {
      throw std::runtime_error("measure: vtable.prepare returned " + std::to_string(rc));
    }
    // release() must run even if measure_launch throws -- a leaked ctx on
    // an error path is a correctness concern for a long-lived process that
    // serves many measurements, not just a cleanliness one.
    struct ReleaseGuard {
      const pmon_family_vtable* vt;
      void* ctx;
      ~ReleaseGuard() { vt->release(ctx); }
    } guard{vtable, ctx};

    perfmon::MeasurementRecord rec = perfmon::measure_launch(*vtable, ctx, stream, entry.seed);
    const char* desc = vtable->describe(ctx);

    std::ostringstream oss;
    oss << "{\"timing_method\":\"" << json_escape(rec.timing_method) << "\""
        << ",\"l2_flush\":" << (rec.l2_flush ? "true" : "false")
        << ",\"n\":" << rec.stats.n
        << ",\"mean\":" << json_double(rec.stats.mean)
        << ",\"median\":" << json_double(rec.stats.median)
        << ",\"stddev\":" << json_double(rec.stats.stddev)
        << ",\"min\":" << json_double(rec.stats.min)
        << ",\"p05\":" << json_double(rec.stats.p05)
        << ",\"p95\":" << json_double(rec.stats.p95)
        << ",\"thermal\":" << json_thermal(rec.thermal)
        << ",\"seed\":" << rec.seed
        << ",\"backend\":" << (desc ? desc : "null")
        << "}";
    return oss.str();
  }

  std::string enumerate_cmd(const std::vector<std::string>& args) {
    // Wire shape: `enumerate <entry_pon> <iface>`.
    if (args.size() < 2) {
      throw std::runtime_error(
          "enumerate: expected at least 2 args (entry_pon iface), got " +
          std::to_string(args.size()));
    }
    const std::string entry_pon = rejoin(args, args.size() - 1);
    const int32_t iface = std::stoi(args[args.size() - 1]);

    perfmon::PmonEntryFields ff = perfmon::parse_entry_pon(entry_pon, iface, subject_id);
    pmon_entry entry{};
    entry.iface = ff.iface;
    entry.dtype = ff.dtype;
    entry.hdim = ff.hdim;
    entry.seqlen_q = ff.seqlen_q;
    entry.seqlen_k = ff.seqlen_k;
    entry.causal = ff.causal;
    entry.dropout_p = ff.dropout_p;
    entry.bias_type = ff.bias_type;
    entry.batch = ff.batch;
    entry.n_heads = ff.n_heads;
    entry.gqa_ratio = ff.gqa_ratio;
    entry.varlen = ff.varlen;
    entry.storage_flip = ff.storage_flip;
    entry.seed = ff.seed;

    constexpr int kMaxBackends = 64;  // every known family reports <= 3
    int backends[kMaxBackends];
    int n = vtable->enumerate_backends(&entry, backends, kMaxBackends);
    if (n < 0) {
      throw std::runtime_error("enumerate: vtable.enumerate_backends returned " +
                                std::to_string(n));
    }
    if (n > kMaxBackends) n = kMaxBackends;  // defensive; should not happen

    std::ostringstream oss;
    oss << "{\"backends\":[";
    for (int i = 0; i < n; ++i) {
      if (i) oss << ",";
      oss << backends[i];
    }
    oss << "]}";
    return oss.str();
  }

  std::string platform(const std::vector<std::string>& /*args*/) {
    perfmon::pmon_thermal t = perfmon::thermal_snapshot();
    std::ostringstream oss;
    oss << "{\"pmon_abi_version\":" << PMON_ABI_VERSION
        << ",\"subject_id\":\"" << json_escape(subject_id) << "\""
        << ",\"thermal\":" << json_thermal(t)
        << "}";
    return oss.str();
  }
};

}  // namespace

int main(int argc, char** argv) {
  const pmon_family_vtable* vtable = pmon_family_entry();
  if (vtable == nullptr) {
    std::cerr << "runner: pmon_family_entry() returned null\n";
    return 1;
  }

  hipStream_t stream = nullptr;
  hipError_t err = hipStreamCreate(&stream);
  if (err != hipSuccess) {
    std::cerr << "runner: hipStreamCreate failed: " << hipGetErrorString(err) << "\n";
    return 1;
  }

  Runner runner;
  runner.vtable = vtable;
  runner.stream = stream;
  runner.subject_id = (argc > 1) ? std::string(argv[1]) : std::string();

  perfmon::Handlers handlers;
  handlers.measure = [&runner](const std::vector<std::string>& args) {
    return runner.measure(args);
  };
  handlers.enumerate_cmd = [&runner](const std::vector<std::string>& args) {
    return runner.enumerate_cmd(args);
  };
  handlers.platform = [&runner](const std::vector<std::string>& args) {
    return runner.platform(args);
  };

  int rc = perfmon::protocol_loop(std::cin, std::cout, handlers);

  hipStreamDestroy(stream);
  return rc;
}
