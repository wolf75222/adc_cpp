// Manifest JSON escape/unescape round-trip for the external-brick registry (Spec 3 section 21-22,
// ADC-463). The host emits each brick's id/category/requirements/capabilities into the JSON manifest
// `pops.lib.load_cpp_library` parses with `json.loads`; the C++ reader (`field`) parses it back. Both
// directions must agree on EVERY byte a user-chosen id/requirement can carry -- structural `"` / `\`
// and any control character -- or a manifest with such a token is silently truncated (C++ side) or
// rejected outright (`json.loads` raises on a raw control char). This is a pure-string test: it does
// NOT need Kokkos or _pops, so it stays a fast, always-on gate independent of the device toolchain.

#include <pops/runtime/program/external_brick.hpp>

#include <cstdio>
#include <string>

using pops::runtime::program::json_escape;
using pops::runtime::program::json_unescape;

namespace {

int fails = 0;

void chk(bool cond, const char* label) {
  std::printf("  [%s] %s\n", cond ? "OK " : "XX ", label);
  if (!cond)
    ++fails;
}

// json_escape then json_unescape recovers the original byte-for-byte.
void roundtrip(const std::string& raw, const char* label) {
  chk(json_unescape(json_escape(raw)) == raw, label);
}

// A valid JSON string body carries no raw control char and no bare `"`; every backslash starts a
// recognized escape. This is the property `json.loads` needs (a raw control char makes it raise).
bool is_valid_json_string_body(const std::string& e) {
  for (std::size_t i = 0; i < e.size(); ++i) {
    const unsigned char c = static_cast<unsigned char>(e[i]);
    if (c < 0x20)
      return false;  // raw control char -> json.loads rejects
    if (c == '"')
      return false;  // bare quote -> ends the string early
    if (c == '\\') {
      if (i + 1 >= e.size())
        return false;
      ++i;  // skip the escaped char (json_escape only emits \" \\ \n \r \t \b \f \uXXXX)
    }
  }
  return true;
}

}  // namespace

int main() {
  // round-trips over the bytes a brick token can carry
  roundtrip("my_riemann", "plain identifier");
  roundtrip("B_z,T_e,rho", "requirements CSV (commas untouched)");
  roundtrip("a\"b", "embedded double quote");
  roundtrip("a\\b", "embedded backslash");
  roundtrip("c:/p\\q", "windows-ish path");
  roundtrip(std::string("line1\nline2\tcol\r"), "newline + tab + carriage return");
  roundtrip(std::string("x\x01y\x1fz", 5), "low control bytes 0x01 / 0x1f");
  roundtrip("", "empty string");

  // the escaped form is always a valid JSON string body (what json.loads requires)
  chk(is_valid_json_string_body(json_escape("a\"b\\c")), "escaped quote+backslash is valid JSON");
  chk(is_valid_json_string_body(json_escape(std::string("n\nt\tctrl\x02", 8))),
      "escaped control chars are valid JSON");

  // a value carrying an escaped quote is NOT truncated at the inner quote (the field-scan contract)
  chk(json_unescape(json_escape("pre\"post")) == "pre\"post", "escaped quote not truncated");

  if (fails == 0)
    std::printf("OK test_external_brick_json\n");
  return fails ? 1 : 0;
}
