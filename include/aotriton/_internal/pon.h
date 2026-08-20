// Copyright © 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef AOTRITON_V2_API_PON_H
#define AOTRITON_V2_API_PON_H

#include <aotriton/config.h>

#include <cstdint>
#include <optional>
#include <string_view>

namespace AOTRITON_NS {

// A borrowed, read-only view over a ';'-separated 'k=v' PON (Plain / Python
// Object Notation) string. Holds no storage: the backing text is the
// compiled-in packed_string, which has static storage duration. Never
// construct from a temporary std::string.
//
// Grammar (measured from the packed strings actually emitted; the Python
// producer is python/utils/pon.py's `render_pon`):
//   section := pair (';' pair)*
//   pair    := key '=' value
//   value   := Python repr -- 0 | -1 | True | False | None | 'transposed' | 'auto'
//
// Every value is a plain `repr()`: a string is always single-quoted with
// nothing escaped (render_pon asserts this at build time), so `get_str` only
// ever needs to strip exactly one leading and one trailing quote.
//
// Parsing uses std::from_chars exclusively (never strtol/atoi/sscanf): the
// backing string_view is not NUL-terminated and may point into the middle of a
// larger concatenated buffer, so any NUL-seeking libc parser risks reading past
// this view's own end into unrelated data. Every lookup here is strictly
// bounded to [text_.begin(), text_.end()).
//
// `None` (any field can render it -- every knob is `T | None`) is treated as a
// parse failure by every accessor, exactly like a missing key: it means "this
// knob was not computed for this image", which `dflt` (or the fatal assert) is
// meant to handle, not a literal three-letter string result.
class Pon {
public:
  constexpr explicit Pon(std::string_view text = {}) noexcept : text_(text) {}

  bool contains(std::string_view key) const noexcept;
  std::optional<std::string_view> find(std::string_view key) const noexcept;

  // Return the parsed value. On a missing key, an unparsable value ("None"),
  // or a range error: return `dflt` if given, else log the key and assert
  // (fatal) -- a silently-zero-filled launch geometry is worse than a crash.
  int64_t          get_int (std::string_view key, std::optional<int64_t> dflt = std::nullopt) const;
  bool             get_bool(std::string_view key, std::optional<bool> dflt = std::nullopt) const;
  std::string_view get_str (std::string_view key, std::optional<std::string_view> dflt = std::nullopt) const;

private:
  std::string_view text_;
};

}

#endif
