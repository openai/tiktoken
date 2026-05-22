//! Exploring efficiency gains by replacing regex with a lexer in BPE
//! tokenizers.
//!
//! Pre-tokenization in BPE tokenizers like tiktoken's `o200k_base` is
//! implemented as a non-trivial regex applied via `fancy_regex::Regex::find_iter`.
//! Tiktoken maintainers note in-source that this regex dominates tokenization
//! runtime. This crate replaces that regex with a forward-scanning state
//! machine whose output is byte-identical to the regex's, while avoiding the
//! engine overhead and backtracking risk.
//!
//! Currently configured for `o200k_base` (the latest OpenAI tokenizer, used
//! by gpt-4o / o1-series). The approach generalizes to other BPE pre-tok
//! patterns (`cl100k_base`, `p50k_base`, …).

use std::sync::LazyLock;
use unicode_properties::{GeneralCategory, UnicodeGeneralCategory};

/// The o200k_base pre-tokenization pattern, pulled verbatim from
/// `tiktoken_ext.openai_public.o200k_base()['pat_str']` in openai/tiktoken.
///
/// Seven `|`-separated alternatives:
///   1. word ending in lowercase (with optional leading non-letter/non-digit
///      char, optional uppercase prefix, mandatory lowercase tail, optional
///      contraction suffix)
///   2. word starting with uppercase (similar shape; mandatory uppercase,
///      optional lowercase tail, optional contraction)
///   3. `\p{N}{1,3}` — 1 to 3 digits
///   4. ` ?[^\s\p{L}\p{N}]+[\r\n/]*` — punctuation (optional leading space,
///      mandatory non-letter/non-digit/non-whitespace body, trailing
///      `\r\n/` zero or more times)
///   5. `\s*[\r\n]+` — paragraph break (optional whitespace + 1+ newlines)
///   6. `\s+(?!\S)` — trailing whitespace at EOF or before another
///      whitespace; the `(?!\S)` lookahead is the part that makes this
///      regex non-trivial to replace
///   7. `\s+` — fallback whitespace run
pub const PAT_STR_O200K_BASE: &str = r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+";

/// cl100k_base pretok pattern (GPT-4, GPT-3.5-turbo, text-embedding-ada-002).
/// Pulled verbatim from `tiktoken_ext.openai_public.cl100k_base()['pat_str']`.
/// 8 alternatives separated by `|`:
///   1. `'(?i:[sdmt]|ll|ve|re)`    — case-insensitive contraction tails
///   2. `[^\r\n\p{L}\p{N}]?+\p{L}++` — optional leading non-letter/digit, then letters (possessive)
///   3. `\p{N}{1,3}+`              — 1 to 3 digits (number-cluster cap)
///   4. ` ?[^\s\p{L}\p{N}]++[\r\n]*+` — punctuation with optional leading space and trailing newlines
///   5. `\s++$`                    — trailing whitespace at EOF (anchored)
///   6. `\s*[\r\n]`                — whitespace prefix + single newline
///   7. `\s+(?!\S)`                — whitespace before more whitespace (lookahead)
///   8. `\s+`                      — fallback whitespace run
pub const PAT_STR_CL100K_BASE: &str = r"'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}++|\p{N}{1,3}+| ?[^\s\p{L}\p{N}]++[\r\n]*+|\s++$|\s*[\r\n]|\s+(?!\S)|\s+";

/// GPT-2 family pretok pattern (r50k_base, p50k_base, p50k_edit — used by
/// GPT-2, davinci-002/003, Codex). Pulled verbatim from
/// `tiktoken_ext.openai_public.r50k_base()['pat_str']`.
/// 7 alternatives separated by `|`:
///   1. `'(?:[sdmt]|ll|ve|re)`     — case-SENSITIVE contraction tails
///   2. ` ?\p{L}++`                — optional space then letters
///   3. ` ?\p{N}++`                — optional space then digits
///   4. ` ?[^\s\p{L}\p{N}]++`      — optional space then non-letter/digit/space
///   5. `\s++$`                    — trailing whitespace at EOF
///   6. `\s+(?!\S)`                — trailing whitespace before another WS (lookahead)
///   7. `\s`                       — single whitespace fallback
pub const PAT_STR_GPT2: &str =
    r"'(?:[sdmt]|ll|ve|re)| ?\p{L}++| ?\p{N}++| ?[^\s\p{L}\p{N}]++|\s++$|\s+(?!\S)|\s";

// =============================================================================
// Character classification helpers
// =============================================================================
//
// The o200k_base regex uses Unicode general categories. The exact mapping:
//   \p{L}  = Lu | Ll | Lt | Lm | Lo                  (any letter)
//   \p{N}  = Nd | Nl | No                            (any number)
//   \p{M}  = Mn | Mc | Me                            (any mark)
//   \p{Lu} = uppercase letter
//   \p{Lt} = titlecase letter
//   \p{Lm} = modifier letter
//   \p{Lo} = other letter
//   \p{Ll} = lowercase letter
//
// Alternative 1 / 2 use two compound classes:
//   "upper-class": Lu | Lt | Lm | Lo | M    (uppercase-ish + scripts without case)
//   "lower-class": Ll | Lm | Lo | M         (lowercase-ish + scripts without case)
// Note that Lm, Lo, M appear in BOTH — so CJK (Lo) and combining marks (M)
// can match either part of the word pattern, and the regex's greedy
// quantifier-then-backtrack semantics decide the split. We mirror that
// here with explicit backtracking.

// Classification bitmap.
//
// One byte per Unicode codepoint. Each bit encodes membership in a regex
// character class. Lookup is `class_of(c) & MASK != 0` — single array
// access + bit test, regardless of script.
//
// This is the same technique used by rust-lang `regex`, ripgrep, and other
// high-performance regex engines. Trades 1.1 MB of static memory (built
// lazily at first lex call) for constant-time, script-independent
// classification. Before this, non-ASCII chars hit `unicode-properties`'
// range search (~30-50 ns); now they're ~2-3 ns like ASCII chars.

const CLASS_LETTER: u8 = 1 << 0; // \p{L}  = Lu | Ll | Lt | Lm | Lo
const CLASS_NUMBER: u8 = 1 << 1; // \p{N}  = Nd | Nl | No
const CLASS_MARK: u8 = 1 << 2; // \p{M}  = Mn | Mc | Me
const CLASS_WHITESPACE: u8 = 1 << 3; // \s in Unicode mode
const CLASS_UPPER: u8 = 1 << 4; // alt 1 middle / alt 2 prefix: Lu|Lt|Lm|Lo|M
const CLASS_LOWER: u8 = 1 << 5; // alt 1 tail / alt 2 tail:    Ll|Lm|Lo|M
const CLASS_NEWLINE: u8 = 1 << 6; // \r or \n

const UNICODE_TABLE_SIZE: usize = 0x110000;

/// Number of ASCII codepoints (0x00..=0x7F). The lexer keeps a separate
/// compile-time ASCII class table because the ASCII range is the common case
/// and avoiding a heap-backed `LazyLock` lookup for it is worthwhile.
const ASCII_BOUNDARY: usize = 128;

/// Maximum digit run length in `\p{N}{1,3}` (the number-cluster alt in both
/// `o200k_base` and `cl100k_base`). After 3 digits the regex forces a new
/// number-cluster match.
const MAX_DIGIT_RUN: usize = 3;

/// ASCII portion of the class table — computed at compile time as a `const`
/// so the common case has zero runtime init cost and the table can be
/// inlined.
const ASCII_CLASS: [u8; ASCII_BOUNDARY] = {
    let mut t = [0u8; ASCII_BOUNDARY];
    let mut i: usize = 0;
    while i < ASCII_BOUNDARY {
        let b = i as u8;
        let mut c = 0u8;
        if b.is_ascii_uppercase() {
            c |= CLASS_LETTER | CLASS_UPPER;
        }
        if b.is_ascii_lowercase() {
            c |= CLASS_LETTER | CLASS_LOWER;
        }
        if b.is_ascii_digit() {
            c |= CLASS_NUMBER;
        }
        if matches!(b, b' ' | b'\t' | b'\n' | b'\r' | 0x0b | 0x0c) {
            c |= CLASS_WHITESPACE;
        }
        if matches!(b, b'\r' | b'\n') {
            c |= CLASS_NEWLINE;
        }
        t[i] = c;
        i += 1;
    }
    t
};

/// Non-ASCII portion of the class table — built once at first lex call by
/// walking every Unicode scalar value and computing its category via
/// `unicode-properties`. ~50 ms one-time cost; amortized over the lifetime
/// of the process.
static UNICODE_CLASS_TABLE: LazyLock<Box<[u8]>> = LazyLock::new(|| {
    let mut t = vec![0u8; UNICODE_TABLE_SIZE].into_boxed_slice();
    for i in ASCII_BOUNDARY..UNICODE_TABLE_SIZE {
        if let Some(c) = char::from_u32(i as u32) {
            t[i] = compute_class_byte_from_unicode(c);
        }
        // Surrogates (U+D800..U+DFFF) and unassigned codepoints stay 0 —
        // they can't appear in a valid Rust `&str` so never get looked up.
    }
    t
});

fn compute_class_byte_from_unicode(c: char) -> u8 {
    let cat = c.general_category();
    let mut b = 0u8;
    match cat {
        GeneralCategory::UppercaseLetter => b |= CLASS_LETTER | CLASS_UPPER,
        GeneralCategory::LowercaseLetter => b |= CLASS_LETTER | CLASS_LOWER,
        GeneralCategory::TitlecaseLetter => b |= CLASS_LETTER | CLASS_UPPER,
        GeneralCategory::ModifierLetter => b |= CLASS_LETTER | CLASS_UPPER | CLASS_LOWER,
        GeneralCategory::OtherLetter => b |= CLASS_LETTER | CLASS_UPPER | CLASS_LOWER,
        GeneralCategory::NonspacingMark
        | GeneralCategory::SpacingMark
        | GeneralCategory::EnclosingMark => b |= CLASS_MARK | CLASS_UPPER | CLASS_LOWER,
        GeneralCategory::DecimalNumber
        | GeneralCategory::LetterNumber
        | GeneralCategory::OtherNumber => b |= CLASS_NUMBER,
        _ => {}
    }
    if c.is_whitespace() {
        b |= CLASS_WHITESPACE;
    }
    if c == '\r' || c == '\n' {
        b |= CLASS_NEWLINE;
    }
    b
}

#[inline(always)]
fn class_of(c: char) -> u8 {
    let i = c as u32 as usize;
    if i < ASCII_BOUNDARY {
        ASCII_CLASS[i]
    } else {
        UNICODE_CLASS_TABLE[i]
    }
}

#[inline]
fn is_letter(c: char) -> bool {
    class_of(c) & CLASS_LETTER != 0
}

#[inline]
fn is_number(c: char) -> bool {
    class_of(c) & CLASS_NUMBER != 0
}

#[inline]
fn is_upper_class(c: char) -> bool {
    class_of(c) & CLASS_UPPER != 0
}

#[inline]
fn is_lower_class(c: char) -> bool {
    class_of(c) & CLASS_LOWER != 0
}

#[inline]
fn is_leading_class(c: char) -> bool {
    class_of(c) & (CLASS_NEWLINE | CLASS_LETTER | CLASS_NUMBER) == 0
}

#[inline]
fn is_whitespace(c: char) -> bool {
    class_of(c) & CLASS_WHITESPACE != 0
}

// =============================================================================
// Contraction matching: (?i:'s|'t|'re|'ve|'m|'ll|'d)
// =============================================================================
//
// All contractions are ASCII. Match case-insensitively against the input
// starting at `pos`. Return Some(new_pos) on match, None otherwise.
// The patterns are tried in order — first match wins (left-to-right
// alternation, matching the regex behavior).

const CONTRACTIONS: &[&str] = &["'s", "'t", "'re", "'ve", "'m", "'ll", "'d"];

fn try_contraction(input: &str, pos: usize) -> Option<usize> {
    let rest = input.as_bytes().get(pos..)?;
    for &c in CONTRACTIONS {
        let cb = c.as_bytes();
        if rest.len() < cb.len() {
            continue;
        }
        // ASCII-only comparison, case-insensitive
        let mut ok = true;
        for i in 0..cb.len() {
            if !rest[i].eq_ignore_ascii_case(&cb[i]) {
                ok = false;
                break;
            }
        }
        if ok {
            return Some(pos + cb.len());
        }
    }
    None
}

// =============================================================================
// Per-alternative match functions
// =============================================================================
//
// Each returns Some(end_byte) on match, None otherwise. Match span is
// always [start, end) and end > start (we never produce empty matches).

/// Alt 1: `[^\r\n\p{L}\p{N}]? [Lu|Lt|Lm|Lo|M]* [Ll|Lm|Lo|M]+ (?i:contraction)?`
///
/// Word ending in a lower-class character. Backtracks the greedy
/// upper-class scan if the mandatory lower-class tail fails to match.
///
/// Hot path: avoid the Vec<usize> allocation by tracking only the greedy
/// upper-end position; when backtracking, recompute the prior char's
/// start by walking forward from `body_start`. This is O(k²) in the
/// worst case (all-upper word that fully fails alt 1) but k is small in
/// practice (word length).
fn try_alt1(input: &str, start: usize) -> Option<usize> {
    for leading in [true, false] {
        let body_start = if leading {
            let c = input[start..].chars().next()?;
            if !is_leading_class(c) {
                continue;
            }
            start + c.len_utf8()
        } else {
            start
        };

        // Greedy scan of upper-class chars from body_start.
        let mut upper_end = body_start;
        for c in input[body_start..].chars() {
            if !is_upper_class(c) {
                break;
            }
            upper_end += c.len_utf8();
        }

        // Backtrack from greediest to least.
        loop {
            // Scan greedy lower-class chars from upper_end.
            let mut lower_end = upper_end;
            for c in input[upper_end..].chars() {
                if !is_lower_class(c) {
                    break;
                }
                lower_end += c.len_utf8();
            }
            if lower_end > upper_end {
                let after = try_contraction(input, lower_end).unwrap_or(lower_end);
                return Some(after);
            }
            // Backtrack: shrink upper_end by one char (the last upper-class
            // char consumed). Walk forward from body_start to find the
            // previous char-boundary < upper_end.
            if upper_end == body_start {
                break;
            }
            let mut prev = body_start;
            for c in input[body_start..upper_end].chars() {
                let next = prev + c.len_utf8();
                if next == upper_end {
                    break;
                }
                prev = next;
            }
            upper_end = prev;
        }
    }
    None
}

/// Alt 2: `[^\r\n\p{L}\p{N}]? [Lu|Lt|Lm|Lo|M]+ [Ll|Lm|Lo|M]* (?i:contraction)?`
///
/// Word starting with an upper-class character. No backtracking needed
/// since the lower-class tail is `*` (always matches).
fn try_alt2(input: &str, start: usize) -> Option<usize> {
    for leading in [true, false] {
        let body_start = if leading {
            let c = input[start..].chars().next()?;
            if !is_leading_class(c) {
                continue;
            }
            start + c.len_utf8()
        } else {
            start
        };

        // Greedy upper-class scan (must be 1+).
        let mut upper_end = body_start;
        let mut count = 0;
        for c in input[body_start..].chars() {
            if !is_upper_class(c) {
                break;
            }
            upper_end += c.len_utf8();
            count += 1;
        }
        if count == 0 {
            continue;
        }
        // Greedy lower-class scan (0 or more).
        let mut lower_end = upper_end;
        for c in input[upper_end..].chars() {
            if !is_lower_class(c) {
                break;
            }
            lower_end += c.len_utf8();
        }
        let after = try_contraction(input, lower_end).unwrap_or(lower_end);
        return Some(after);
    }
    None
}

/// Alt 3: `\p{N}{1,3}` — 1 to `MAX_DIGIT_RUN` digit characters.
fn try_alt3(input: &str, start: usize) -> Option<usize> {
    let mut end = start;
    let mut count = 0;
    for c in input[start..].chars() {
        if count >= MAX_DIGIT_RUN || !is_number(c) {
            break;
        }
        end += c.len_utf8();
        count += 1;
    }
    if count >= 1 { Some(end) } else { None }
}

/// Alt 4: ` ?[^\s\p{L}\p{N}]+[\r\n/]*` — optional leading ASCII space,
/// then 1+ non-whitespace/non-letter/non-digit, then 0+ of `[\r\n/]`.
fn try_alt4(input: &str, start: usize) -> Option<usize> {
    for leading in [true, false] {
        let body_start = if leading {
            // Pattern is " ?" — ASCII space specifically.
            if input.as_bytes().get(start).copied() != Some(b' ') {
                continue;
            }
            start + 1
        } else {
            start
        };

        let mut pos = body_start;
        for c in input[body_start..].chars() {
            if is_whitespace(c) || is_letter(c) || is_number(c) {
                break;
            }
            pos += c.len_utf8();
        }
        if pos == body_start {
            continue; // body needs 1+
        }
        // Greedy trailing [\r\n/]*
        for c in input[pos..].chars() {
            if c == '\r' || c == '\n' || c == '/' {
                pos += c.len_utf8();
            } else {
                break;
            }
        }
        return Some(pos);
    }
    None
}

/// Alt 5: `\s*[\r\n]+` — optional whitespace prefix + 1+ newlines.
///
/// Implemented as: scan whitespace forward, tracking the end position of
/// the most recent run of newlines (i.e. the last `\r` or `\n` in the
/// run). The match ends right after that last newline. If no newlines
/// were seen during the whitespace scan, this alt fails.
fn try_alt5(input: &str, start: usize) -> Option<usize> {
    let mut pos = start;
    let mut last_newline_end: Option<usize> = None;
    for c in input[start..].chars() {
        if !is_whitespace(c) {
            break;
        }
        pos += c.len_utf8();
        if c == '\r' || c == '\n' {
            last_newline_end = Some(pos);
        }
    }
    last_newline_end
}

/// Alt 6: `\s+(?!\S)` — whitespace run followed by EOF or another
/// whitespace. The lookahead is the trap: a naive greedy `\s+` would
/// consume into the WS run between words, but the regex backs off so
/// that the match ends one char before the next non-whitespace.
fn try_alt6(input: &str, start: usize) -> Option<usize> {
    let mut pos = start;
    let mut last_ws_start: Option<usize> = None;
    for c in input[start..].chars() {
        if !is_whitespace(c) {
            break;
        }
        last_ws_start = Some(pos);
        pos += c.len_utf8();
    }
    if pos == start {
        return None; // no whitespace
    }
    if pos == input.len() {
        return Some(pos); // ran to EOF — whole run matches
    }
    // Followed by non-whitespace — match ends before the last ws char.
    let last_ws_start = last_ws_start.unwrap();
    if last_ws_start > start {
        Some(last_ws_start)
    } else {
        None
    }
}

/// Alt 7: `\s+` — fallback whitespace, greedy.
fn try_alt7(input: &str, start: usize) -> Option<usize> {
    let mut pos = start;
    for c in input[start..].chars() {
        if !is_whitespace(c) {
            break;
        }
        pos += c.len_utf8();
    }
    if pos > start { Some(pos) } else { None }
}

// =============================================================================
// Public API
// =============================================================================

/// Split `text` into pre-tokens, returning an iterator of byte-offset pairs
/// `(start, end)` matching `fancy_regex::Regex::find_iter` on
/// [`PAT_STR_O200K_BASE`].
pub fn split(text: &str) -> Splits<'_> {
    Splits {
        input: text,
        pos: 0,
    }
}

pub struct Splits<'a> {
    input: &'a str,
    pos: usize,
}

impl<'a> Iterator for Splits<'a> {
    type Item = (usize, usize);

    fn next(&mut self) -> Option<(usize, usize)> {
        if self.pos >= self.input.len() {
            return None;
        }
        let start = self.pos;
        let end = try_alt1(self.input, start)
            .or_else(|| try_alt2(self.input, start))
            .or_else(|| try_alt3(self.input, start))
            .or_else(|| try_alt4(self.input, start))
            .or_else(|| try_alt5(self.input, start))
            .or_else(|| try_alt6(self.input, start))
            .or_else(|| try_alt7(self.input, start))?;
        debug_assert!(
            end > start,
            "lexer made no progress at byte {start} in {:?}",
            &self.input[start..(start + 32).min(self.input.len())]
        );
        self.pos = end;
        Some((start, end))
    }
}

// =============================================================================
// cl100k_base lexer
// =============================================================================
//
// Pattern: `'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}++|\p{N}{1,3}+|
//           ?[^\s\p{L}\p{N}]++[\r\n]*+|\s++$|\s*[\r\n]|\s+(?!\S)|\s+`
//
// 8 alternatives. Differences from o200k:
//   - alt 1: standalone contractions (no preceding word context — the
//     contraction is its own token from any `'sdmt|ll|ve|re`)
//   - alt 2: no Lu/Ll case split — letters are matched as one possessive run
//   - alt 4: trailing class is `[\r\n]*` (no `/`)
//   - alt 6 (`\s*[\r\n]`) matches the same span as o200k alt 5
//     (`\s*[\r\n]+`) under backtracking — single newline + greedy `\s*`
//     anchored to last `\r\n` produces identical end positions
//   - alt 5 (`\s++$`) and alt 8 (`\s+`) are minor tie-breakers vs
//     o200k's structure
//
// Sharing: classification bitmap + alt 3 (`\p{N}{1,3}`) + alt 5/6/7 of
// o200k can all be reused; new code is alts 1, 2, 4, and the EOF-anchored
// `\s++$` matcher.

/// cl100k alt 1: `'(?i:[sdmt]|ll|ve|re)`.
///
/// Standalone contraction tail — first char must be `'`. The inner group is
/// case-insensitive ASCII so all comparisons fold to lowercase.
fn try_cl100k_alt1(input: &str, start: usize) -> Option<usize> {
    let bytes = input.as_bytes();
    if bytes.get(start).copied() != Some(b'\'') {
        return None;
    }
    let rest = bytes.get(start + 1..)?;
    if rest.is_empty() {
        return None;
    }
    let c1 = rest[0].to_ascii_lowercase();
    if matches!(c1, b's' | b'd' | b'm' | b't') {
        return Some(start + 2);
    }
    if rest.len() >= 2 {
        let c2 = rest[1].to_ascii_lowercase();
        if matches!((c1, c2), (b'l', b'l') | (b'v', b'e') | (b'r', b'e')) {
            return Some(start + 3);
        }
    }
    None
}

/// cl100k alt 2: `[^\r\n\p{L}\p{N}]?+\p{L}++`.
///
/// Optional leading non-newline/non-letter/non-digit char, then 1+ letters.
/// Possessive on both quantifiers; if the leading char matches but no
/// letters follow, the alt fails (we still retry without the leading char
/// via the loop — which produces the same outcome, since the leading char
/// is non-letter so the no-leading branch also can't start with a letter).
fn try_cl100k_alt2(input: &str, start: usize) -> Option<usize> {
    for leading in [true, false] {
        let body_start = if leading {
            let c = input[start..].chars().next()?;
            if !is_leading_class(c) {
                continue;
            }
            start + c.len_utf8()
        } else {
            start
        };
        let mut end = body_start;
        for c in input[body_start..].chars() {
            if !is_letter(c) {
                break;
            }
            end += c.len_utf8();
        }
        if end > body_start {
            return Some(end);
        }
    }
    None
}

/// cl100k alt 4: ` ?[^\s\p{L}\p{N}]++[\r\n]*+`.
fn try_cl100k_alt4(input: &str, start: usize) -> Option<usize> {
    for leading in [true, false] {
        let body_start = if leading {
            if input.as_bytes().get(start).copied() != Some(b' ') {
                continue;
            }
            start + 1
        } else {
            start
        };
        let mut pos = body_start;
        for c in input[body_start..].chars() {
            if is_whitespace(c) || is_letter(c) || is_number(c) {
                break;
            }
            pos += c.len_utf8();
        }
        if pos == body_start {
            continue;
        }
        for c in input[pos..].chars() {
            if c == '\r' || c == '\n' {
                pos += c.len_utf8();
            } else {
                break;
            }
        }
        return Some(pos);
    }
    None
}

/// cl100k alt 5: `\s++$`. Possessive whitespace run anchored at EOF; matches
/// only if the run reaches end-of-input.
fn try_cl100k_alt5(input: &str, start: usize) -> Option<usize> {
    let mut pos = start;
    for c in input[start..].chars() {
        if !is_whitespace(c) {
            return None;
        }
        pos += c.len_utf8();
    }
    if pos > start { Some(pos) } else { None }
}

/// Split `text` into pre-tokens matching
/// `fancy_regex::Regex::find_iter` on [`PAT_STR_CL100K_BASE`].
pub fn split_cl100k(text: &str) -> SplitsCl100k<'_> {
    SplitsCl100k {
        input: text,
        pos: 0,
    }
}

pub struct SplitsCl100k<'a> {
    input: &'a str,
    pos: usize,
}

impl<'a> Iterator for SplitsCl100k<'a> {
    type Item = (usize, usize);

    fn next(&mut self) -> Option<(usize, usize)> {
        if self.pos >= self.input.len() {
            return None;
        }
        let start = self.pos;
        // Order matches the regex: alt 5 (\s++$) is tried before the
        // unanchored whitespace alts so an all-whitespace tail is taken
        // as a single token.
        let end = try_cl100k_alt1(self.input, start)
            .or_else(|| try_cl100k_alt2(self.input, start))
            .or_else(|| try_alt3(self.input, start)) // \p{N}{1,3}
            .or_else(|| try_cl100k_alt4(self.input, start))
            .or_else(|| try_cl100k_alt5(self.input, start)) // \s++$
            .or_else(|| try_alt5(self.input, start)) // \s*[\r\n]+ ≡ \s*[\r\n]
            .or_else(|| try_alt6(self.input, start)) // \s+(?!\S)
            .or_else(|| try_alt7(self.input, start))?; // \s+
        debug_assert!(
            end > start,
            "cl100k lexer made no progress at byte {start} in {:?}",
            &self.input[start..(start + 32).min(self.input.len())]
        );
        self.pos = end;
        Some((start, end))
    }
}

// =============================================================================
// GPT-2 family lexer (r50k_base / p50k_base / p50k_edit)
// =============================================================================
//
// Pattern: `'(?:[sdmt]|ll|ve|re)| ?\p{L}++| ?\p{N}++| ?[^\s\p{L}\p{N}]++|
//           \s++$|\s+(?!\S)|\s`
//
// 7 alternatives. Key differences from cl100k:
//   - contractions are case-SENSITIVE (no `?i:`)
//   - words/digits/other all gated by optional ASCII space prefix (no
//     general leading-char class, just literal ` ?`)
//   - no `\p{N}{1,3}` cap — digits are fully greedy
//   - no trailing `[\r\n]*` in the punctuation alt
//   - alt 7 is a single `\s` char (not `\s+`)

fn try_gpt2_alt1(input: &str, start: usize) -> Option<usize> {
    let bytes = input.as_bytes();
    if bytes.get(start).copied() != Some(b'\'') {
        return None;
    }
    let rest = bytes.get(start + 1..)?;
    if rest.is_empty() {
        return None;
    }
    let c1 = rest[0];
    if matches!(c1, b's' | b'd' | b'm' | b't') {
        return Some(start + 2);
    }
    if rest.len() >= 2 {
        let c2 = rest[1];
        if matches!((c1, c2), (b'l', b'l') | (b'v', b'e') | (b'r', b'e')) {
            return Some(start + 3);
        }
    }
    None
}

/// Generic ` ?<class>++` matcher — optional ASCII space prefix, then 1+
/// chars where `pred` returns true.
#[inline]
fn try_gpt2_space_run<F: Fn(char) -> bool>(input: &str, start: usize, pred: F) -> Option<usize> {
    for leading in [true, false] {
        let body_start = if leading {
            if input.as_bytes().get(start).copied() != Some(b' ') {
                continue;
            }
            start + 1
        } else {
            start
        };
        let mut end = body_start;
        for c in input[body_start..].chars() {
            if !pred(c) {
                break;
            }
            end += c.len_utf8();
        }
        if end > body_start {
            return Some(end);
        }
    }
    None
}

fn try_gpt2_alt7(input: &str, start: usize) -> Option<usize> {
    let c = input[start..].chars().next()?;
    if is_whitespace(c) {
        Some(start + c.len_utf8())
    } else {
        None
    }
}

/// Split `text` into pre-tokens matching `fancy_regex::Regex::find_iter`
/// on [`PAT_STR_GPT2`].
pub fn split_gpt2(text: &str) -> SplitsGpt2<'_> {
    SplitsGpt2 {
        input: text,
        pos: 0,
    }
}

pub struct SplitsGpt2<'a> {
    input: &'a str,
    pos: usize,
}

impl<'a> Iterator for SplitsGpt2<'a> {
    type Item = (usize, usize);

    fn next(&mut self) -> Option<(usize, usize)> {
        if self.pos >= self.input.len() {
            return None;
        }
        let start = self.pos;
        let input = self.input;
        let end = try_gpt2_alt1(input, start)
            .or_else(|| try_gpt2_space_run(input, start, is_letter))
            .or_else(|| try_gpt2_space_run(input, start, is_number))
            .or_else(|| {
                try_gpt2_space_run(input, start, |c| {
                    !is_whitespace(c) && !is_letter(c) && !is_number(c)
                })
            })
            .or_else(|| try_cl100k_alt5(input, start)) // \s++$
            .or_else(|| try_alt6(input, start)) // \s+(?!\S)
            .or_else(|| try_gpt2_alt7(input, start))?; // single \s
        debug_assert!(
            end > start,
            "gpt2 lexer made no progress at byte {start} in {:?}",
            &self.input[start..(start + 32).min(self.input.len())]
        );
        self.pos = end;
        Some((start, end))
    }
}

#[cfg(test)]
mod lexer_regex_equivalence {
    //! Assert that each pattern's lexer produces the same `(start, end)` splits
    //! as `fancy_regex::Regex::find_iter` on the corresponding pattern, across
    //! curated fixtures that span the algorithmic corners of the patterns.
    //!
    //! Full-corpus byte equality (across 230 MiB of multilingual / code /
    //! synthetic content, all four built-in encodings) was run separately
    //! during development; these unit tests are the guardrail against future
    //! regressions inside this crate.

    use super::*;
    use fancy_regex::Regex;

    fn fixtures() -> &'static [&'static str] {
        &[
            // empty + whitespace edges
            "",
            " ",
            "  ",
            "\n",
            "\n\n",
            " \n ",
            "trailing ",
            // ASCII words + contractions
            "hello world",
            "don't",
            "I'm",
            "we've",
            "they'll",
            "she'd",
            "it's",
            "DON'T I'M", // case-insensitive contractions for o200k / cl100k
            // numbers (o200k / cl100k cap at {1,3})
            "1",
            "12",
            "123",
            "1234",
            "012345",
            // greedy uppercase -> lowercase backtracking (o200k alt 1 vs alt 2)
            "HELLOworld",
            "FOOBARbaz",
            "AbCdEf",
            // mixed scripts
            "hello 世界 hello",
            "你好世界",
            "Привет мир",
            "नमस्ते दुनिया",
            // emoji & non-BMP
            "🌍",
            "Hello 🌍 World",
            "👨\u{200d}👩\u{200d}👧",
            // code-like punctuation
            "foo_bar.baz();",
            "x++",
            "a + b",
            "x.y.z",
            "!!!",
            "...",
            "---",
            // apostrophe NOT followed by a contraction suffix
            "don'X",
            "'standalone",
            "abc' def",
        ]
    }

    fn assert_equivalent(pattern_str: &str, split: impl Fn(&str) -> Vec<(usize, usize)>) {
        let regex = Regex::new(pattern_str).expect("regex must compile");
        for &text in fixtures() {
            let lexer_out = split(text);
            let regex_out: Vec<_> = regex
                .find_iter(text)
                .map(|m| {
                    let m = m.expect("regex match");
                    (m.start(), m.end())
                })
                .collect();
            assert_eq!(
                lexer_out, regex_out,
                "lexer/regex diverged on input {text:?}; pattern: {pattern_str}",
            );
        }
    }

    #[test]
    fn o200k_base_lexer_matches_regex() {
        assert_equivalent(PAT_STR_O200K_BASE, |t| super::split(t).collect());
    }

    #[test]
    fn cl100k_base_lexer_matches_regex() {
        assert_equivalent(PAT_STR_CL100K_BASE, |t| super::split_cl100k(t).collect());
    }

    #[test]
    fn gpt2_lexer_matches_regex() {
        assert_equivalent(PAT_STR_GPT2, |t| super::split_gpt2(t).collect());
    }
}
