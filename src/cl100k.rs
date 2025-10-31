use std::cmp::min;

use unicode_properties::{GeneralCategory, UnicodeGeneralCategory};

pub const CL100K_PATTERN: &str = r"'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}++|\p{N}{1,3}+| ?[^\s\p{L}\p{N}]++[\r\n]*+|\s++$|\s*[\r\n]|\s+(?!\S)|\s";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Cl100kMatchKind {
    Contraction,
    LetterWithPrefix,
    Number,
    Punctuation,
    WhitespaceToEof,
    WhitespaceThenLinebreak,
    TrailingWhitespace,
    SingleWhitespace,
    Fallback,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct Cl100kParser;

impl Cl100kParser {
    pub fn new() -> Self {
        Self
    }

    pub fn find_iter<'a>(&self, text: &'a str) -> Cl100kMatches<'a> {
        Cl100kMatches { text, offset: 0 }
    }
}

#[derive(Debug, Clone)]
pub struct Cl100kMatch<'a> {
    haystack: &'a str,
    start: usize,
    end: usize,
    kind: Cl100kMatchKind,
}

impl<'a> Cl100kMatch<'a> {
    pub fn as_str(&self) -> &'a str {
        &self.haystack[self.start..self.end]
    }

    pub fn start(&self) -> usize {
        self.start
    }

    pub fn end(&self) -> usize {
        self.end
    }

    pub fn len(&self) -> usize {
        self.end - self.start
    }

    pub fn kind(&self) -> Cl100kMatchKind {
        self.kind
    }
}

pub struct Cl100kMatches<'a> {
    text: &'a str,
    offset: usize,
}

impl<'a> Iterator for Cl100kMatches<'a> {
    type Item = Cl100kMatch<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.offset >= self.text.len() {
            return None;
        }

        let start = self.offset;
        let slice = self.text;

        let (advance, kind) = match_branch(slice, start).unwrap_or_else(|| {
            let next = char_at(slice, start)
                .map(|(_, next)| next)
                .unwrap_or_else(|| min(start + 1, slice.len()));
            (next - start, Cl100kMatchKind::Fallback)
        });

        let end = start + advance;
        self.offset = end;

        Some(Cl100kMatch {
            haystack: self.text,
            start,
            end,
            kind,
        })
    }
}

fn match_branch(text: &str, idx: usize) -> Option<(usize, Cl100kMatchKind)> {
    match_contraction(text, idx)
        .or_else(|| match_word_with_optional_prefix(text, idx))
        .or_else(|| match_short_number(text, idx))
        .or_else(|| match_punct_run(text, idx))
        .or_else(|| match_whitespace_to_eof(text, idx))
        .or_else(|| match_ws_then_linebreak(text, idx))
        .or_else(|| match_trailing_ws(text, idx))
        .or_else(|| match_single_ws(text, idx))
}

// Regex branch: `'(?i:[sdmt]|ll|ve|re)`
fn match_contraction(text: &str, idx: usize) -> Option<(usize, Cl100kMatchKind)> {
    let (first, next) = char_at(text, idx)?;
    if first != '\'' {
        return None;
    }

    let (a, after_a) = char_at(text, next)?;
    let lower_a = ascii_lower(a);

    if matches!(lower_a, 's' | 'd' | 'm' | 't') {
        return Some((after_a - idx, Cl100kMatchKind::Contraction));
    }

    let (b, after_b) = char_at(text, after_a)?;
    let lower_b = ascii_lower(b);

    if (lower_a == 'l' && lower_b == 'l')
        || (lower_a == 'v' && lower_b == 'e')
        || (lower_a == 'r' && lower_b == 'e')
    {
        return Some((after_b - idx, Cl100kMatchKind::Contraction));
    }

    None
}

// Regex branch: `[^\r\n\p{L}\p{N}]?+\p{L}++`
fn match_word_with_optional_prefix(text: &str, idx: usize) -> Option<(usize, Cl100kMatchKind)> {
    let (cp0, next0) = char_at(text, idx)?;

    if !is_cr_or_lf(cp0) && !is_alnum(cp0) {
        let (cp1, _) = char_at(text, next0)?;
        if !is_letter(cp1) {
            return None;
        }
        let end = consume_letters(text, next0)?;
        return Some((end - idx, Cl100kMatchKind::LetterWithPrefix));
    }

    if !is_letter(cp0) {
        return None;
    }

    let end = consume_letters(text, idx)?;
    Some((end - idx, Cl100kMatchKind::LetterWithPrefix))
}

fn consume_letters(text: &str, start: usize) -> Option<usize> {
    let mut end = start;
    let mut count = 0usize;
    while let Some((ch, next)) = char_at(text, end) {
        if !is_letter(ch) {
            break;
        }
        end = next;
        count += 1;
    }
    if count == 0 { None } else { Some(end) }
}

// Regex branch: `\p{N}{1,3}+`
fn match_short_number(text: &str, idx: usize) -> Option<(usize, Cl100kMatchKind)> {
    let (cp1, mut end) = char_at(text, idx)?;
    if !is_number(cp1) {
        return None;
    }

    let mut count = 1usize;
    while count < 3 {
        if let Some((cp, next)) = char_at(text, end) {
            if is_number(cp) {
                end = next;
                count += 1;
            } else {
                break;
            }
        } else {
            break;
        }
    }

    Some((end - idx, Cl100kMatchKind::Number))
}

// Regex branch: ` ?[^\s\p{L}\p{N}]++[\r\n]*+`
fn match_punct_run(text: &str, idx: usize) -> Option<(usize, Cl100kMatchKind)> {
    let mut cursor = idx;

    if let Some((ch, next)) = char_at(text, cursor) {
        if ch == ' ' {
            let (after_space, _) = char_at(text, next)?;
            if is_space(after_space) || is_alnum(after_space) {
                return None;
            }
            cursor = next;
        }
    }

    let mut end = cursor;
    let mut took = false;
    while let Some((ch, next)) = char_at(text, end) {
        if is_space(ch) || is_alnum(ch) {
            break;
        }
        end = next;
        took = true;
    }

    if !took {
        return None;
    }

    while let Some((ch, next)) = char_at(text, end) {
        if !is_cr_or_lf(ch) {
            break;
        }
        end = next;
    }

    Some((end - idx, Cl100kMatchKind::Punctuation))
}

// Regex branch: `\s++$`
fn match_whitespace_to_eof(text: &str, idx: usize) -> Option<(usize, Cl100kMatchKind)> {
    let (first, mut end) = char_at(text, idx)?;
    if !is_space(first) {
        return None;
    }

    while let Some((ch, next)) = char_at(text, end) {
        if !is_space(ch) {
            break;
        }
        end = next;
    }

    if end == text.len() {
        Some((end - idx, Cl100kMatchKind::WhitespaceToEof))
    } else {
        None
    }
}

// Regex branch: `\s*[\r\n]`
fn match_ws_then_linebreak(text: &str, idx: usize) -> Option<(usize, Cl100kMatchKind)> {
    let mut pos = idx;
    let mut best: Option<usize> = None;

    if let Some((ch, _)) = char_at(text, pos) {
        if is_cr_or_lf(ch) {
            best = Some(pos);
        }
    }

    while let Some((ch, next)) = char_at(text, pos) {
        if !is_space(ch) {
            break;
        }
        pos = next;
        if let Some((next_ch, _)) = char_at(text, pos) {
            if is_cr_or_lf(next_ch) {
                best = Some(pos);
            }
        }
    }

    let newline_pos = best?;
    let (_, newline_end) = char_at(text, newline_pos)?;
    Some((newline_end - idx, Cl100kMatchKind::WhitespaceThenLinebreak))
}

// Regex branch: `\s+(?!\S)`
fn match_trailing_ws(text: &str, idx: usize) -> Option<(usize, Cl100kMatchKind)> {
    let (first, mut run_end) = char_at(text, idx)?;
    if !is_space(first) {
        return None;
    }

    while let Some((ch, next)) = char_at(text, run_end) {
        if !is_space(ch) {
            break;
        }
        run_end = next;
    }

    if run_end == text.len() {
        return Some((run_end - idx, Cl100kMatchKind::TrailingWhitespace));
    }

    let prev_start = prev_char_start(text, run_end, idx)?;
    if prev_start == idx {
        return None;
    }

    Some((prev_start - idx, Cl100kMatchKind::TrailingWhitespace))
}

// Regex branch: `\s`
fn match_single_ws(text: &str, idx: usize) -> Option<(usize, Cl100kMatchKind)> {
    let (ch, next) = char_at(text, idx)?;
    if is_space(ch) {
        Some((next - idx, Cl100kMatchKind::SingleWhitespace))
    } else {
        None
    }
}

fn char_at(text: &str, idx: usize) -> Option<(char, usize)> {
    if idx >= text.len() {
        return None;
    }
    let mut iter = text[idx..].char_indices();
    let (offset, ch) = iter.next()?;
    let next = idx + offset + ch.len_utf8();
    Some((ch, next))
}

fn prev_char_start(text: &str, idx: usize, floor: usize) -> Option<usize> {
    if idx <= floor {
        return None;
    }
    let slice = &text[floor..idx];
    slice
        .char_indices()
        .last()
        .map(|(offset, _)| floor + offset)
}

fn ascii_lower(c: char) -> char {
    if c.is_ascii_uppercase() {
        c.to_ascii_lowercase()
    } else {
        c
    }
}

fn is_cr_or_lf(ch: char) -> bool {
    matches!(ch, '\r' | '\n')
}

fn is_letter(ch: char) -> bool {
    matches!(
        ch.general_category(),
        GeneralCategory::UppercaseLetter
            | GeneralCategory::LowercaseLetter
            | GeneralCategory::TitlecaseLetter
            | GeneralCategory::ModifierLetter
            | GeneralCategory::OtherLetter
    ) && ch.is_alphabetic()
}

fn is_number(ch: char) -> bool {
    matches!(
        ch.general_category(),
        GeneralCategory::DecimalNumber
            | GeneralCategory::LetterNumber
            | GeneralCategory::OtherNumber
    ) && ch.is_numeric()
}

fn is_space(ch: char) -> bool {
    matches!(
        ch.general_category(),
        GeneralCategory::SpaceSeparator
            | GeneralCategory::LineSeparator
            | GeneralCategory::ParagraphSeparator
    ) || ch.is_whitespace()
}

fn is_alnum(ch: char) -> bool {
    is_letter(ch) || is_number(ch)
}

#[cfg(test)]
mod tests {
    use super::*;
    use unicode_properties::UnicodeGeneralCategory;

    #[test]
    fn contraction_matches() {
        let parser = Cl100kParser::new();
        let matches = parser
            .find_iter("'re")
            .map(|m| m.as_str().to_string())
            .collect::<Vec<_>>();
        assert_eq!(matches, vec!["'re".to_string()]);
    }

    #[test]
    fn optional_prefix_word() {
        let parser = Cl100kParser::new();
        let mut iter = parser.find_iter("!Hello world");
        let first = iter.next().unwrap();
        assert_eq!(first.as_str(), "!Hello");
        assert_eq!(first.kind(), Cl100kMatchKind::LetterWithPrefix);
    }

    #[test]
    fn numeric_span_limits_to_three_digits() {
        let parser = Cl100kParser::new();
        let mut iter = parser.find_iter("1234");
        let first = iter.next().unwrap();
        assert_eq!(first.as_str(), "123");
        assert_eq!(first.kind(), Cl100kMatchKind::Number);
        let second = iter.next().unwrap();
        assert_eq!(second.as_str(), "4");
        assert_eq!(second.kind(), Cl100kMatchKind::Number);
    }

    #[test]
    fn punctuation_run_consumes_trailing_newlines() {
        let parser = Cl100kParser::new();
        let mut iter = parser.find_iter(" !?\nfoo");
        let first = iter.next().unwrap();
        assert_eq!(first.as_str(), " !?\n");
        assert_eq!(first.kind(), Cl100kMatchKind::Punctuation);
    }

    #[test]
    fn whitespace_to_eof_branch() {
        let parser = Cl100kParser::new();
        let mut iter = parser.find_iter("foo   ");
        let _ = iter.next();
        let spatial = iter.next().unwrap();
        assert_eq!(spatial.as_str(), "   ");
        assert_eq!(spatial.kind(), Cl100kMatchKind::WhitespaceToEof);
    }

    #[test]
    fn whitespace_then_linebreak_branch() {
        let parser = Cl100kParser::new();
        let mut iter = parser.find_iter("  \nabc");
        let first = iter.next().unwrap();
        assert_eq!(first.as_str(), "  \n");
        assert_eq!(first.kind(), Cl100kMatchKind::WhitespaceThenLinebreak);
    }

    #[test]
    fn trailing_whitespace_branch() {
        let parser = Cl100kParser::new();
        let mut iter = parser.find_iter("  X");
        let first = iter.next().unwrap();
        assert_eq!(first.as_str(), " ");
        assert_eq!(first.kind(), Cl100kMatchKind::TrailingWhitespace);
    }

    #[test]
    fn single_whitespace_branch() {
        let parser = Cl100kParser::new();
        let mut iter = parser.find_iter("\t!");
        let first = iter.next().unwrap();
        assert_eq!(first.as_str(), "\t");
        assert_eq!(first.kind(), Cl100kMatchKind::SingleWhitespace);
    }

    #[test]
    fn debug_fancy_letter_sample() {
        let ch = '\u{323B0}';
        let fancy = fancy_regex::Regex::new(r"\p{L}").unwrap();
        let fancy_match = fancy.is_match(&ch.to_string()).unwrap();
        println!(
            "char U+323B0 general_category={:?} fancy_match={}",
            ch.general_category(),
            fancy_match
        );
        println!(
            "char U+323B0 is_alphabetic={} is_alphanumeric={} is_whitespace={}",
            ch.is_alphabetic(),
            ch.is_alphanumeric(),
            ch.is_whitespace()
        );
        assert_eq!(is_letter(ch), fancy_match);
    }
}
