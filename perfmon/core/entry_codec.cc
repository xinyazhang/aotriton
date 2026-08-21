// Copyright © 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "entry_codec.h"

#include <cctype>
#include <map>
#include <stdexcept>
#include <string_view>
#include <utility>

namespace perfmon {

namespace {

std::string trim(std::string_view sv) {
  size_t b = 0, e = sv.size();
  while (b < e && std::isspace(static_cast<unsigned char>(sv[b]))) ++b;
  while (e > b && std::isspace(static_cast<unsigned char>(sv[e - 1]))) --e;
  return std::string(sv.substr(b, e - b));
}

// Splits a ';'-separated `k=v` PON string into a raw (unparsed-value) map.
// First occurrence of a duplicate key wins, matching the reference reader's
// (`Pon::find`, personal/flyc-codegen-rewrite:include/aotriton/_internal/
// pon.h) documented "first wins" rule -- kept consistent even though this
// file is a separate, scoped-down implementation (see entry_codec.h's top
// comment for why that other reader was not reused directly).
std::map<std::string, std::string> split_pairs(const std::string& text) {
  std::map<std::string, std::string> out;
  size_t pos = 0;
  while (pos <= text.size()) {
    size_t semi = text.find(';', pos);
    std::string pair = (semi == std::string::npos) ? text.substr(pos)
                                                     : text.substr(pos, semi - pos);
    size_t eq = pair.find('=');
    if (eq != std::string::npos) {
      std::string key = trim(std::string_view(pair).substr(0, eq));
      std::string val = trim(std::string_view(pair).substr(eq + 1));
      out.emplace(std::move(key), std::move(val));
    }
    if (semi == std::string::npos) break;
    pos = semi + 1;
  }
  return out;
}

[[noreturn]] void fail(const std::string& key, const std::string& why) {
  throw std::invalid_argument("entry_codec: key \"" + key + "\": " + why);
}

const std::string& require(const std::map<std::string, std::string>& d, const std::string& key) {
  auto it = d.find(key);
  if (it == d.end()) fail(key, "missing from entry_pon");
  return it->second;
}

// render_pon (python/utils/pon.py) emits every str value as a plain
// single-quoted repr() with nothing escaped (asserted at build time on the
// Python side) -- recovering the original is always this two-character
// strip, never a general repr unescaper. Same contract the reference
// `Pon::get_str` documents.
std::string parse_pon_str(const std::string& key, const std::string& raw) {
  if (raw.size() >= 2 && raw.front() == '\'' && raw.back() == '\'') {
    return raw.substr(1, raw.size() - 2);
  }
  fail(key, "expected a single-quoted string, got \"" + raw + "\"");
}

bool parse_pon_bool(const std::string& key, const std::string& raw) {
  if (raw == "True") return true;
  if (raw == "False") return false;
  fail(key, "expected True/False, got \"" + raw + "\"");
}

double parse_pon_double(const std::string& key, const std::string& raw) {
  try {
    size_t consumed = 0;
    double v = std::stod(raw, &consumed);
    if (consumed != raw.size()) fail(key, "trailing garbage after number \"" + raw + "\"");
    return v;
  } catch (const std::exception&) {
    fail(key, "expected a number, got \"" + raw + "\"");
  }
}

int64_t parse_pon_int(const std::string& key, const std::string& raw) {
  try {
    size_t consumed = 0;
    long long v = std::stoll(raw, &consumed);
    if (consumed != raw.size()) fail(key, "trailing garbage after integer \"" + raw + "\"");
    return static_cast<int64_t>(v);
  } catch (const std::exception&) {
    fail(key, "expected an integer, got \"" + raw + "\"");
  }
}

// N_HEADS is `int | tuple[int, int]` on FlashEntry/FlashInputMetadata
// (modules/flash/tune/entry.py) -- a plain int for a non-GQA entry, or
// `(num_q_heads, num_kv_heads)` for GQA (desc.py's
// `dataclasses.replace(im, N_HEADS=(10, 2))` convention, mirrored by
// modules/flash/perfmon/entry.py's `_GQA_N_HEADS = (10, 2)`). Handles
// exactly these two shapes; anything else (e.g. a >2-tuple, which nothing
// in this codebase currently produces) fails loudly rather than guessing.
//
// Returns {n_heads, gqa_ratio}.
std::pair<int32_t, int32_t> parse_n_heads(const std::string& raw) {
  const std::string key = "N_HEADS";
  if (!raw.empty() && raw.front() == '(') {
    if (raw.back() != ')') fail(key, "unterminated tuple \"" + raw + "\"");
    std::string inner = raw.substr(1, raw.size() - 2);
    size_t comma = inner.find(',');
    if (comma == std::string::npos) fail(key, "expected a 2-tuple, got \"" + raw + "\"");
    std::string a = trim(inner.substr(0, comma));
    std::string rest = inner.substr(comma + 1);
    if (rest.find(',') != std::string::npos) {
      fail(key, "only 2-tuples are supported, got \"" + raw + "\"");
    }
    std::string b = trim(rest);
    int64_t q = parse_pon_int(key, a);
    int64_t kv = parse_pon_int(key, b);
    if (kv <= 0 || q <= 0 || q % kv != 0) {
      fail(key, "GQA tuple (" + a + ", " + b + ") must have kv>0, q>0, q%kv==0");
    }
    return {static_cast<int32_t>(q), static_cast<int32_t>(q / kv)};
  }
  int64_t n = parse_pon_int(key, raw);
  return {static_cast<int32_t>(n), 1};
}

}  // namespace

int32_t dtype_name_to_pmon_dtype(const std::string& name) {
  // perfmon_abi.h's `enum pmon_dtype`: FLOAT16=0, BFLOAT16=1, FLOAT32=2.
  // Values transcribed here (this file cannot include the HIP-dependent
  // perfmon_abi.h) -- kept in sync by hand; see entry_codec.h's top comment.
  if (name == "float16") return 0;
  if (name == "bfloat16") return 1;
  if (name == "float32") return 2;
  fail("dtype", "unrecognized dtype name \"" + name + "\"");
}

uint64_t fnv1a64(const std::string& s) {
  uint64_t h = 0xcbf29ce484222325ULL;  // FNV offset basis
  for (unsigned char c : s) {
    h ^= c;
    h *= 0x100000001b3ULL;  // FNV prime
  }
  return h;
}

PmonEntryFields parse_entry_pon(const std::string& entry_pon, int32_t iface,
                                 const std::string& subject_id) {
  auto d = split_pairs(entry_pon);

  PmonEntryFields f;
  f.iface = iface;  // see entry_codec.h item 1 -- never read from the PON text
  f.dtype = dtype_name_to_pmon_dtype(parse_pon_str("dtype", require(d, "dtype")));
  f.hdim = static_cast<int32_t>(parse_pon_int("hdim", require(d, "hdim")));
  f.seqlen_q = static_cast<int32_t>(parse_pon_int("seqlen_q", require(d, "seqlen_q")));
  f.seqlen_k = static_cast<int32_t>(parse_pon_int("seqlen_k", require(d, "seqlen_k")));
  f.causal = parse_pon_bool("causal", require(d, "causal")) ? 1 : 0;
  f.dropout_p = parse_pon_double("dropout_p", require(d, "dropout_p"));
  f.bias_type = static_cast<int32_t>(parse_pon_int("bias_type", require(d, "bias_type")));
  f.batch = static_cast<int32_t>(parse_pon_int("BATCH", require(d, "BATCH")));
  {
    auto [n_heads, gqa_ratio] = parse_n_heads(require(d, "N_HEADS"));
    f.n_heads = n_heads;
    f.gqa_ratio = gqa_ratio;
  }
  f.varlen = parse_pon_bool("varlen", require(d, "varlen")) ? 1 : 0;
  f.storage_flip = parse_pon_bool("storage_flip", require(d, "storage_flip")) ? 1 : 0;

  // rev0 §5.3: seed = hash(functional_pon, shape_pon, subject, iface). See
  // entry_codec.h item 3 for why this is derived here rather than read from
  // the wire.
  std::string seed_input = entry_pon + "|" + subject_id + "|" + std::to_string(iface);
  f.seed = fnv1a64(seed_input);

  return f;
}

}  // namespace perfmon
