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
            'é', '中', '٢', 'Ⅳ', '\u{301}', '\u{85}', '\u{a0}', '\u{2003}', '\u{2028}', '\u{202f}',
            '🦀',
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
