// Copyright © 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <aotriton/_internal/schemaless.h>
#include <aotriton/_internal/log.h>

#include <charconv>
#include <cstdio>
#include <cstdlib>

namespace AOTRITON_NS {

namespace {

// Find the value substring of the FIRST 'key=value' pair matching `key` in
// `text` (';'-separated). Duplicate keys: first wins, per PLAN-PHASE2.md Task
// 2. Matches on the FULL key -- 'block_m' must not match a pair keyed
// 'block_mask'. Never reads outside [text.begin(), text.end()); if `text` is a
// sub-view deliberately cut out of a larger buffer (no NUL at its own end),
// this still only ever inspects text[0, text.size()).
std::optional<std::string_view> find_value(std::string_view text, std::string_view key) noexcept {
  size_t pos = 0;
  while (pos <= text.size()) {
    size_t semi = text.find(';', pos);
    std::string_view pair = (semi == std::string_view::npos)
                             ? text.substr(pos)
                             : text.substr(pos, semi - pos);
    size_t eq = pair.find('=');
    if (eq != std::string_view::npos) {
      std::string_view pkey = pair.substr(0, eq);
      if (pkey == key) {
        return pair.substr(eq + 1);
      }
    }
    if (semi == std::string_view::npos)
      break;
    pos = semi + 1;
  }
  return std::nullopt;
}

// Unconditional (not gated by AOTRITON_DEBUG_LEVEL) fatal report: a missing or
// unparsable Schemaless key with no `dflt` is a generator/description bug, not
// a routine diagnostic, and NDEBUG release builds turn a bare assert() into a
// no-op -- which would let execution fall through to a silently wrong launch
// geometry, exactly the failure mode this class exists to avoid. So this
// always prints to stderr and aborts, regardless of build type or log level.
[[noreturn]] void fatal(const char* accessor, std::string_view key) {
  std::fprintf(stderr,
               "[FATAL] Schemaless::%s: key \"%.*s\" is missing, unparsable, "
               "or None, and no default was given\n",
               accessor, int(key.size()), key.data());
  std::abort();
}

}  // anonymous namespace

bool Schemaless::contains(std::string_view key) const noexcept {
  return find_value(text_, key).has_value();
}

std::optional<std::string_view> Schemaless::find(std::string_view key) const noexcept {
  return find_value(text_, key);
}

int64_t Schemaless::get_int(std::string_view key, std::optional<int64_t> dflt) const {
  auto v = find_value(text_, key);
  if (v) {
    int64_t out = 0;
    const char* begin = v->data();
    const char* end = v->data() + v->size();
    auto [ptr, ec] = std::from_chars(begin, end, out);
    // Bounded, no trailing garbage, overflow (result_out_of_range) is a miss.
    if (ec == std::errc() && ptr == end)
      return out;
  }
  if (dflt)
    return *dflt;
  fatal("get_int", key);
}

bool Schemaless::get_bool(std::string_view key, std::optional<bool> dflt) const {
  auto v = find_value(text_, key);
  if (v) {
    // Exact, case-sensitive match on Python's True/False -- not "1"/"0", not
    // case-insensitive. A typo (e.g. "true") is a miss, not silently accepted.
    if (*v == "True")
      return true;
    if (*v == "False")
      return false;
  }
  if (dflt)
    return *dflt;
  fatal("get_bool", key);
}

std::string_view Schemaless::get_str(std::string_view key, std::optional<std::string_view> dflt) const {
  auto v = find_value(text_, key);
  if (v && *v != "None")
    return *v;
  if (dflt)
    return *dflt;
  fatal("get_str", key);
}

}
