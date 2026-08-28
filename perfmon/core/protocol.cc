// Copyright © 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "protocol.h"

#include <cassert>
#include <sstream>

namespace perfmon {

Command parse_command(const std::string& line) {
  Command cmd;
  std::istringstream iss(line);
  std::string tok;
  bool first = true;
  while (iss >> tok) {
    if (first) {
      cmd.name = tok;
      first = false;
    } else {
      cmd.args.push_back(tok);
    }
  }
  return cmd;
}

void emit_ok(std::ostream& out, const std::string& json_body) {
  out << "OK " << json_body << "\n";
  out.flush();
}

void emit_error(std::ostream& out, const std::string& detail) {
  // Must not be misread as success or as a progress line -- see
  // protocol.h's emit_error doc comment.
  assert(detail.rfind("OK", 0) != 0 &&
         "emit_error: detail must not start with \"OK\"");
  assert(detail.rfind("OVERHEATING:", 0) != 0 &&
         "emit_error: detail must not start with \"OVERHEATING:\"");
  out << detail << "\n";
  out.flush();
}

void emit_overheating(std::ostream& out, const std::string& detail) {
  out << "OVERHEATING: " << detail << "\n";
  out.flush();
}

int protocol_loop(std::istream& in, std::ostream& out, const Handlers& handlers) {
  std::string line;
  while (std::getline(in, line)) {
    if (!line.empty() && line.back() == '\r') {
      line.pop_back();  // tolerate CRLF input, harmless on LF-only streams
    }
    Command cmd = parse_command(line);
    if (cmd.name.empty()) {
      continue;  // blank line: not a protocol violation, just ignored
    }
    if (cmd.name == "exit") {
      return 0;
    }
    try {
      std::string result;
      if (cmd.name == "measure") {
        if (!handlers.measure) {
          throw std::runtime_error("ERROR: no measure handler installed");
        }
        result = handlers.measure(cmd.args);
      } else if (cmd.name == "enumerate") {
        if (!handlers.enumerate_cmd) {
          throw std::runtime_error("ERROR: no enumerate handler installed");
        }
        result = handlers.enumerate_cmd(cmd.args);
      } else if (cmd.name == "platform") {
        if (!handlers.platform) {
          throw std::runtime_error("ERROR: no platform handler installed");
        }
        result = handlers.platform(cmd.args);
      } else {
        throw std::runtime_error("ERROR: unknown command: " + cmd.name);
      }
      emit_ok(out, result);
    } catch (const std::exception& e) {
      emit_error(out, std::string("ERROR: ") + e.what());
    }
  }
  return 0;
}

}  // namespace perfmon
