use fancy_regex::Regex;

pub(crate) const GPT2_PATTERN: &str =
    r"'(?:[sdmt]|ll|ve|re)| ?\p{L}++| ?\p{N}++| ?[^\s\p{L}\p{N}]++|\s++$|\s+(?!\S)|\s";

const CL100K_PATTERN: &str = r"'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}++|\p{N}{1,3}+| ?[^\s\p{L}\p{N}]++[\r\n]*+|\s++$|\s*[\r\n]|\s+(?!\S)|\s";
const O200K_PATTERN: &str = concat!(
    r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|",
    r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|",
    r"\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+",
);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Pattern {
    Gpt2,
    Cl100k,
    O200k,
}

impl Pattern {
    pub(crate) fn from_str(pattern: &str) -> Option<Self> {
        match pattern {
            GPT2_PATTERN => Some(Self::Gpt2),
            CL100K_PATTERN => Some(Self::Cl100k),
            O200K_PATTERN => Some(Self::O200k),
            _ => None,
        }
    }
}

/// Scan ASCII pieces directly, retaining the regex's Unicode semantics at boundaries.
pub(crate) struct Pieces<'a, 'r> {
    text: &'a str,
    regex: &'r Regex,
    pattern: Pattern,
    pos: usize,
}

#[inline]
fn is_space(byte: u8) -> bool {
    matches!(byte, b'\t'..=b'\r' | b' ')
}

#[inline]
fn category(byte: u8) -> u8 {
    if byte.is_ascii_alphabetic() {
        0
    } else if byte.is_ascii_digit() {
        1
    } else if is_space(byte) {
        2
    } else {
        3
    }
}

#[inline]
fn is_newline(byte: u8) -> bool {
    matches!(byte, b'\r' | b'\n')
}

#[inline]
fn contraction_len(bytes: &[u8]) -> usize {
    if bytes.first() != Some(&b'\'') || bytes.len() < 2 {
        return 0;
    }
    let first = bytes[1].to_ascii_lowercase();
    if matches!(first, b's' | b'd' | b'm' | b't') {
        return 2;
    }
    if bytes.len() >= 3
        && matches!(
            (first, bytes[2].to_ascii_lowercase()),
            (b'l', b'l') | (b'v', b'e') | (b'r', b'e')
        )
    {
        return 3;
    }
    0
}

impl<'a, 'r> Pieces<'a, 'r> {
    pub(crate) fn new(text: &'a str, regex: &'r Regex, pattern: Pattern) -> Self {
        Self {
            text,
            regex,
            pattern,
            pos: 0,
        }
    }

    #[inline]
    fn ascii_end(&self) -> Option<usize> {
        let bytes = self.text.as_bytes();
        let start = self.pos;
        let first = bytes[start];
        if !first.is_ascii() {
            return None;
        }
        if self.pattern != Pattern::Gpt2 {
            return self.modern_end(bytes);
        }
        if first == b'\'' {
            let rest = &bytes[start + 1..];
            if matches!(rest.first(), Some(b's' | b'd' | b'm' | b't')) {
                return Some(start + 2);
            }
            if rest.starts_with(b"ll") || rest.starts_with(b"ve") || rest.starts_with(b"re") {
                return Some(start + 3);
            }
        }

        let body = start + usize::from(first == b' ' && start + 1 < bytes.len());
        if !bytes[body].is_ascii() {
            return None;
        }
        let kind = category(bytes[body]);
        if kind != 2 {
            let mut end = body + 1;
            while end < bytes.len() {
                let byte = bytes[end];
                // Unicode may continue this piece, so restart it with the reference regex.
                if !byte.is_ascii() {
                    return None;
                }
                if category(byte) != kind {
                    break;
                }
                end += 1;
            }
            return Some(end);
        }

        let mut end = start + 1;
        while end < bytes.len() && is_space(bytes[end]) {
            end += 1;
        }
        if end < bytes.len() {
            if !bytes[end].is_ascii() {
                return None;
            }
            // Before non-whitespace, \s+(?!\S) leaves the last whitespace character.
            // It may become the optional space prefix of the next piece.
            if end - start > 1 {
                end -= 1;
            }
        }
        Some(end)
    }
    #[inline]
    fn modern_end(&self, bytes: &[u8]) -> Option<usize> {
        let start = self.pos;
        if self.pattern == Pattern::Cl100k {
            let contraction = contraction_len(&bytes[start..]);
            if contraction > 0 {
                return Some(start + contraction);
            }
        }

        // The word alternatives allow one non-letter/number prefix other than CR or LF.
        let prefix = !bytes[start].is_ascii_alphanumeric() && !is_newline(bytes[start]);
        let body = start + usize::from(prefix);
        if body < bytes.len() {
            if !bytes[body].is_ascii() {
                return None;
            }
            if bytes[body].is_ascii_alphabetic() {
                let mut end = body;
                if self.pattern == Pattern::Cl100k {
                    while end < bytes.len() && bytes[end].is_ascii_alphabetic() {
                        end += 1;
                    }
                } else {
                    // Both o200k alternatives consume uppercase* lowercase* in ASCII,
                    // splitting at an uppercase letter after the first lowercase run.
                    while end < bytes.len() && bytes[end].is_ascii_uppercase() {
                        end += 1;
                    }
                    while end < bytes.len() && bytes[end].is_ascii_lowercase() {
                        end += 1;
                    }
                }
                if end < bytes.len() && !bytes[end].is_ascii() {
                    return None;
                }
                if self.pattern == Pattern::O200k && bytes.get(end) == Some(&b'\'') {
                    // Unicode case folding can recognize LONG S in a contraction.
                    let suffix = &bytes[end..bytes.len().min(end + 3)];
                    if !suffix.is_ascii() {
                        return None;
                    }
                    end += contraction_len(&bytes[end..]);
                }
                return Some(end);
            }
        }

        if bytes[start].is_ascii_digit() {
            let mut end = start + 1;
            while end < bytes.len() && end - start < 3 && bytes[end].is_ascii_digit() {
                end += 1;
            }
            if end - start < 3 && end < bytes.len() && !bytes[end].is_ascii() {
                return None;
            }
            return Some(end);
        }

        let body = start + usize::from(bytes[start] == b' ');
        if body < bytes.len() && category(bytes[body]) == 3 {
            let mut end = body + 1;
            while end < bytes.len() {
                if !bytes[end].is_ascii() {
                    return None;
                }
                if category(bytes[end]) != 3 {
                    break;
                }
                end += 1;
            }
            while end < bytes.len()
                && (is_newline(bytes[end])
                    || (self.pattern == Pattern::O200k && bytes[end] == b'/'))
            {
                end += 1;
            }
            return Some(end);
        }

        let mut end = start;
        let mut last_newline = None;
        while end < bytes.len() && is_space(bytes[end]) {
            end += 1;
            if is_newline(bytes[end - 1]) {
                last_newline = Some(end);
            }
        }
        if end < bytes.len() && !bytes[end].is_ascii() {
            return None;
        }
        // cl100k's end-of-input whitespace alternative precedes its newline alternative.
        if self.pattern == Pattern::Cl100k && end == bytes.len() {
            return Some(end);
        }
        if let Some(end) = last_newline {
            return Some(end);
        }
        // Leave the last whitespace character for the next piece before non-whitespace.
        if end < bytes.len() && end - start > 1 {
            end -= 1;
        }
        Some(end)
    }
}

impl<'a> Iterator for Pieces<'a, '_> {
    type Item = Result<&'a [u8], fancy_regex::Error>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.pos == self.text.len() {
            return None;
        }
        let start = self.pos;
        let end = match self.ascii_end() {
            Some(end) => end,
            None => match self.regex.find_from_pos(self.text, start) {
                Ok(Some(mat)) => mat.end(),
                Ok(None) => unreachable!("Built-in patterns cover every character"),
                Err(err) => {
                    self.pos = self.text.len();
                    return Some(Err(err));
                }
            },
        };
        debug_assert!(end > start);
        self.pos = end;
        Some(Ok(&self.text.as_bytes()[start..end]))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn patterns() -> [(Pattern, Regex); 3] {
        [
            (Pattern::Gpt2, GPT2_PATTERN),
            (Pattern::Cl100k, CL100K_PATTERN),
            (Pattern::O200k, O200K_PATTERN),
        ]
        .map(|(pattern, regex)| (pattern, Regex::new(regex).unwrap()))
    }

    fn check(text: &str, pattern: Pattern, regex: &Regex) {
        let expected: Vec<_> = regex
            .find_iter(text)
            .map(|m| m.unwrap().as_str().as_bytes())
            .collect();
        let actual: Vec<_> = Pieces::new(text, regex, pattern)
            .map(Result::unwrap)
            .collect();
        assert_eq!(actual, expected, "{pattern:?}: {text:?}");
    }

    #[test]
    fn ascii_pairs_and_contractions() {
        for (pattern, regex) in patterns() {
            for a in 0..128u8 {
                for b in 0..128u8 {
                    check(std::str::from_utf8(&[a, b]).unwrap(), pattern, &regex);
                }
            }
            for prefix in [
                "",
                " ",
                "  ",
                "\n",
                "\r\n",
                "\t ",
                "!",
                "word",
                "WORD",
                "camelCase",
                "HTTPServer",
            ] {
                for suffix in [
                    "'s", "'t", "'re", "'ve", "'m", "'ll", "'d", "'S", "'RE", "'x", "'ſ", "'K",
                ] {
                    check(&format!("{prefix}{suffix}é 123!\n/  "), pattern, &regex);
                }
            }
            for text in [
                "aA",
                "Aa",
                "aaAAaa",
                "HTTPSConnectionPool",
                "f oo",
                "1234567",
                "a\n  ",
                "\n \n ",
                "\n\t \r\n",
                " !\n/",
                "'rÉ",
                "it'ſ",
                "it'Sé",
                "A\u{301}b",
                "Aa\u{301}B",
            ] {
                check(text, pattern, &regex);
            }
        }
    }

    #[test]
    fn unicode_boundaries() {
        for (pattern, regex) in patterns() {
            for ch in [
                'é', '中', '٢', 'Ⅳ', '\u{301}', '\u{85}', '\u{a0}', '\u{2003}', '\u{2028}',
                '\u{202f}', '🦀',
            ] {
                for a in 0..128u8 {
                    for prefix in ["", " ", "\n ", "word", "WORD"] {
                        check(
                            &format!("{prefix}{}{ch}{}", a as char, a as char),
                            pattern,
                            &regex,
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn randomized_pieces() {
        let alphabet: Vec<_> =
            "aaAZ09 'strvelmd!/_\r\n\t\x0b\x0cé中٢Ⅳ\u{301}\u{85}\u{a0}\u{2003}\u{2028}🦀"
                .chars()
                .collect();
        for (pattern, regex) in patterns() {
            let mut state = 42u64;
            for len in 0..128 {
                for _ in 0..100 {
                    let text: String = (0..len)
                        .map(|_| {
                            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                            alphabet[(state >> 32) as usize % alphabet.len()]
                        })
                        .collect();
                    check(&text, pattern, &regex);
                }
            }
        }
    }
}
