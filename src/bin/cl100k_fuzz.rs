use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::env;
use std::time::{Duration, Instant};

use base64::engine::general_purpose::STANDARD as BASE64;
use base64::Engine;

use rand::distributions::{Distribution, Uniform};
use rand::{rngs::StdRng, Rng, SeedableRng};
use tiktoken::cl100k::{Cl100kMatchKind, Cl100kParser, CL100K_PATTERN};
use tiktoken::{CoreBPE, PatternBackendChoice, Rank};

const DEFAULT_STEPS: usize = 5_000;
const DEFAULT_MAX_LEN: usize = 512;

#[derive(Clone, Debug, PartialEq, Eq)]
enum Mode {
    Split,
    Bpe,
    File(String),
}

const INTERESTING_CHARS: &[char] = &[
    '\0', '\u{0001}', '\u{0002}', '\u{0003}', '\u{0004}', '\u{0005}', '\u{0006}', '\u{0007}',
    '\u{0008}', '\t', '\n', '\u{000B}', '\u{000C}', '\r', '\u{000E}', '\u{000F}', '\u{0010}',
    '\u{0011}', '\u{0012}', '\u{0013}', '\u{0014}', '\u{0015}', '\u{0016}', '\u{0017}', '\u{0018}',
    '\u{0019}', '\u{001A}', '\u{001B}', '\u{001C}', '\u{001D}', '\u{001E}', '\u{001F}', ' ',
    '\u{0085}', '\u{00A0}', '\u{1680}', '\u{180E}', '\u{2000}', '\u{2001}', '\u{2002}', '\u{2003}',
    '\u{2004}', '\u{2005}', '\u{2006}', '\u{2007}', '\u{2008}', '\u{2009}', '\u{200A}', '\u{2028}',
    '\u{2029}', '\u{202F}', '\u{205F}', '\u{3000}', '\u{FEFF}',
];

const ASCII_WHITESPACE: &[char] = &[' ', '\t', '\n', '\r', '\u{000B}', '\u{000C}'];
const ASCII_ALNUM: &[char] = &[
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S',
    'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l',
    'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '0', '1', '2', '3', '4',
    '5', '6', '7', '8', '9',
];
const ASCII_PUNCT: &[char] = &[
    '!', '"', '#', '$', '%', '&', '\'', '(', ')', '*', '+', ' ', ',', '-', '.', '/', ':', ';', '<',
    '=', '>', '?', '@', '[', '\\', ']', '^', '_', '`', '{', '|', '}', '~',
];

#[derive(Debug, Clone, PartialEq, Eq)]
struct CustomSpan {
    start: usize,
    end: usize,
    kind: Cl100kMatchKind,
}

fn main() {
    let (mode, steps, max_len) = parse_args();
    println!("Starting cl100k fuzz: mode={:?}, steps={steps}, max_len={max_len}", mode);

    match mode {
        Mode::Split => run_split_fuzz(steps, max_len),
        Mode::Bpe => run_bpe_fuzz(steps, max_len),
        Mode::File(path) => run_file_bench(&path, steps),
    }
}

fn parse_args() -> (Mode, usize, usize) {
    let mut args = env::args().skip(1);
    let mode = match args.next().as_deref() {
        Some("split") | None => Mode::Split,
        Some("bpe") => Mode::Bpe,
        Some("file") => {
            let path = args.next().unwrap_or_else(|| {
                eprintln!("Usage: cl100k_fuzz file <input_path>");
                std::process::exit(2);
            });
            Mode::File(path)
        }
        Some(other) => {
            eprintln!("Unknown mode '{other}'. Use 'split', 'bpe', or 'file'.");
            std::process::exit(2);
        }
    };
    let steps = args
        .next()
        .as_deref()
        .map(parse_usize)
        .unwrap_or(DEFAULT_STEPS);
    let max_len = args
        .next()
        .as_deref()
        .map(parse_usize)
        .unwrap_or(DEFAULT_MAX_LEN);
    if steps == 0 {
        eprintln!("Step count must be greater than zero");
        std::process::exit(2);
    }
    (mode, steps, max_len)
}

fn parse_usize(arg: &str) -> usize {
    arg.parse().unwrap_or_else(|_| {
        eprintln!("Invalid integer argument: {arg}");
        std::process::exit(2);
    })
}

fn generate_input(rng: &mut StdRng, max_len: usize) -> String {
    let len_range = Uniform::new_inclusive(0, max_len);
    let len = len_range.sample(rng);
    let mut s = String::with_capacity(len);
    for _ in 0..len {
        s.push(sample_char(rng));
    }
    s
}
fn run_split_fuzz(steps: usize, max_len: usize) {
    use std::io::Write;

    let parser = Cl100kParser::new();
    let mut rng = StdRng::from_entropy();

    let mut fancy_total = Duration::ZERO;
    let mut custom_total = Duration::ZERO;

    for iter in 1..=steps {
        let input = generate_input(&mut rng, max_len);

        let fancy_start = Instant::now();
        let fancy = collect_fancy_spans(&input);
        fancy_total += fancy_start.elapsed();

        let custom_start = Instant::now();
        let custom = collect_custom_spans(&input, &parser);
        custom_total += custom_start.elapsed();

        if let Err(err) = compare_spans(&input, &fancy, &custom) {
            eprintln!("\nMismatch detected on iteration {iter}");
            eprintln!("Input ({} bytes):", input.len());
            dump_bytes(&input);
            eprintln!("{err}");
            std::process::exit(1);
        }

        if iter % 100 == 0 {
            print!("\rCompleted {iter}/{steps} cases");
            let _ = std::io::stdout().flush();
        }
    }

    println!("\nFinished {steps} cases with no mismatches.");
    println!("Total fancy_regex split time: {:?}", fancy_total);
    println!("Total custom parser split time: {:?}", custom_total);
    let fancy_avg = fancy_total.as_secs_f64() / steps as f64;
    let custom_avg = custom_total.as_secs_f64() / steps as f64;
    println!(
        "Average per case: fancy_regex={:.6}s, custom_parser={:.6}s",
        fancy_avg, custom_avg
    );
    if fancy_total.is_zero() || custom_total.is_zero() {
        println!("Custom speedup vs Fancy: n/a");
    } else {
        println!(
            "Custom speedup vs Fancy: {:.3}x",
            fancy_total.as_secs_f64() / custom_total.as_secs_f64()
        );
    }
}

fn run_bpe_fuzz(steps: usize, max_len: usize) {
    use std::io::Write;

    let (encoder, specials) = load_bpe_from_file("../cl100k_base.tiktoken")
        .unwrap_or_else(|e| {
            eprintln!("Failed to load cl100k_base.tiktoken: {e}");
            std::process::exit(2);
        });
    let decoder = make_decoder_map(&encoder);
    let bpe_custom = CoreBPE::new_with_backend::<_, _, std::iter::Empty<(String, (Rank, Rank))>>(
        encoder.clone(),
        specials.clone(),
        CL100K_PATTERN,
        PatternBackendChoice::Cl100kParser,
    )
    .expect("failed to construct custom CoreBPE");
    let bpe_fancy = CoreBPE::new_with_backend::<_, _, std::iter::Empty<(String, (Rank, Rank))>>(
        encoder,
        specials,
        CL100K_PATTERN,
        PatternBackendChoice::FancyRegex,
    )
    .expect("failed to construct fancy CoreBPE");

    let mut rng = StdRng::from_entropy();

    let mut fancy_total = Duration::ZERO;
    let mut custom_total = Duration::ZERO;

    for iter in 1..=steps {
        let input = generate_input(&mut rng, max_len);

        let t0 = Instant::now();
        let fancy_tokens = bpe_fancy.encode_ordinary(&input);
        fancy_total += t0.elapsed();

        let t1 = Instant::now();
        let custom_tokens = bpe_custom.encode_ordinary(&input);
        custom_total += t1.elapsed();

        if fancy_tokens != custom_tokens {
            eprintln!("\nToken mismatch on iteration {iter}");
            eprintln!("Input ({} bytes):", input.len());
            dump_bytes(&input);
            eprintln!("fancy tokens (len={}):", fancy_tokens.len());
            dump_tokens(&fancy_tokens);
            eprintln!("custom tokens (len={}):", custom_tokens.len());
            dump_tokens(&custom_tokens);
            std::process::exit(1);
        }
        // if iter % 500 == 0 {
        //     let fancy_dec = decode_tokens(&fancy_tokens, &decoder);
        //     let custom_dec = decode_tokens(&custom_tokens, &decoder);
        //     let fancy_text = String::from_utf8_lossy(&fancy_dec);
        //     let custom_text = String::from_utf8_lossy(&custom_dec);
        //     println!("\n[iter {iter}] sample input: {}", preview(&input, 160));
        //     println!("[iter {iter}] fancy decode: {}", preview(&fancy_text, 160));
        //     println!("[iter {iter}] custom decode: {}", preview(&custom_text, 160));
        // }

        if iter % 100 == 0 {
            print!("\rCompleted {iter}/{steps} cases");
            let _ = std::io::stdout().flush();
        }
    }

    println!("\nFinished {steps} cases with no mismatches.");
    println!("Total fancy_regex encode time: {:?}", fancy_total);
    println!("Total custom parser encode time: {:?}", custom_total);
    let fancy_avg = fancy_total.as_secs_f64() / steps as f64;
    let custom_avg = custom_total.as_secs_f64() / steps as f64;
    println!(
        "Average per case: fancy_regex={:.6}s, custom_parser={:.6}s",
        fancy_avg, custom_avg
    );
    if fancy_total.is_zero() || custom_total.is_zero() {
        println!("Custom speedup vs Fancy: n/a");
    } else {
        println!(
            "Custom speedup vs Fancy: {:.3}x",
            fancy_total.as_secs_f64() / custom_total.as_secs_f64()
        );
    }
}

fn run_file_bench(path: &str, iterations: usize) {
    let (encoder, specials) = load_bpe_from_file("../cl100k_base.tiktoken").unwrap_or_else(|e| {
        eprintln!("Failed to load cl100k_base.tiktoken: {e}");
        std::process::exit(2);
    });
    let decoder = make_decoder_map(&encoder);

    let bpe_custom = CoreBPE::new_with_backend::<_, _, std::iter::Empty<(String, (Rank, Rank))>>(
        encoder.clone(),
        specials.clone(),
        CL100K_PATTERN,
        PatternBackendChoice::Cl100kParser,
    )
    .expect("failed to construct custom CoreBPE");
    let bpe_fancy = CoreBPE::new_with_backend::<_, _, std::iter::Empty<(String, (Rank, Rank))>>(
        encoder,
        specials,
        CL100K_PATTERN,
        PatternBackendChoice::FancyRegex,
    )
    .expect("failed to construct fancy CoreBPE");

    let input = std::fs::read_to_string(path).unwrap_or_else(|e| {
        eprintln!("Failed to read input file '{path}': {e}");
        std::process::exit(2);
    });

    // First pass: verify equality once and time single-run (warmup)
    let t0 = Instant::now();
    let fancy_tokens = bpe_fancy.encode_ordinary(&input);
    let fancy_first = t0.elapsed();

    let t1 = Instant::now();
    let custom_tokens = bpe_custom.encode_ordinary(&input);
    let custom_first = t1.elapsed();

    if fancy_tokens != custom_tokens {
        eprintln!("Token mismatch between fancy and custom on file '{}':", path);
        eprintln!("fancy tokens (len={}):", fancy_tokens.len());
        dump_tokens(&fancy_tokens);
        eprintln!("custom tokens (len={}):", custom_tokens.len());
        dump_tokens(&custom_tokens);
        std::process::exit(1);
    }

    let fancy_dec = decode_tokens(&fancy_tokens, &decoder);
    let custom_dec = decode_tokens(&custom_tokens, &decoder);
    if fancy_dec != custom_dec {
        eprintln!("Decoded byte mismatch between fancy and custom on file '{}'.", path);
        std::process::exit(1);
    }

    // Optional: also check roundtrip
    if fancy_dec != input.as_bytes() {
        eprintln!("Warning: decoded bytes differ from input file bytes (possibly due to encoding/normalization). Continuing.");
    }

    // Iterated benchmark over the dataset
    let mut fancy_total = fancy_first; // include first pass
    let mut custom_total = custom_first; // include first pass
    for _ in 1..iterations { // already did one
        let t0 = Instant::now();
        let f = bpe_fancy.encode_ordinary(&input);
        fancy_total += t0.elapsed();

        let t1 = Instant::now();
        let c = bpe_custom.encode_ordinary(&input);
        custom_total += t1.elapsed();
    }

    println!("\nFile bench for '{path}' (iterations={}):", iterations);
    println!("Fancy total:   {:?}", fancy_total);
    println!("Custom total:  {:?}", custom_total);
    let fancy_avg = fancy_total.as_secs_f64() / iterations as f64;
    let custom_avg = custom_total.as_secs_f64() / iterations as f64;
    println!("Average/iter:  fancy={:.6}s custom={:.6}s", fancy_avg, custom_avg);
    if fancy_total.is_zero() || custom_total.is_zero() {
        println!("Custom speedup vs Fancy: n/a");
    } else {
        println!(
            "Custom speedup vs Fancy: {:.3}x",
            fancy_total.as_secs_f64() / custom_total.as_secs_f64()
        );
    }
}

fn load_bpe_from_file(path: &str) -> Result<(HashMap<Vec<u8>, Rank>, HashMap<String, Rank>), String> {
    let file = File::open(path).map_err(|e| e.to_string())?;
    let reader = BufReader::new(file);
    let mut encoder: HashMap<Vec<u8>, Rank> = HashMap::with_capacity(100_000);
    for (lineno, line_res) in reader.lines().enumerate() {
        let line = line_res.map_err(|e| e.to_string())?;
        if line.is_empty() {
            continue;
        }
        let mut parts = line.splitn(2, ' ');
        let tok_b64 = parts.next().ok_or_else(|| format!("Malformed line {}", lineno + 1))?;
        let rank_str = parts.next().ok_or_else(|| format!("Malformed line {}", lineno + 1))?;
        let bytes = BASE64
            .decode(tok_b64.as_bytes())
            .map_err(|e| format!("base64 decode error at line {}: {}", lineno + 1, e))?;
        let rank: Rank = rank_str
            .parse::<u32>()
            .map_err(|e| format!("rank parse error at line {}: {}", lineno + 1, e))?;
        encoder.insert(bytes, rank);
    }
    let specials: HashMap<String, Rank> = HashMap::new();
    Ok((encoder, specials))
}

fn make_decoder_map(encoder: &HashMap<Vec<u8>, Rank>) -> HashMap<Rank, Vec<u8>> {
    let mut decoder: HashMap<Rank, Vec<u8>> = HashMap::with_capacity(encoder.len());
    for (bytes, &rank) in encoder.iter() {
        decoder.insert(rank, bytes.clone());
    }
    decoder
}

fn decode_tokens(tokens: &[Rank], decoder: &HashMap<Rank, Vec<u8>>) -> Vec<u8> {
    let mut out = Vec::with_capacity(tokens.len() * 2);
    for &t in tokens {
        if let Some(b) = decoder.get(&t) {
            out.extend_from_slice(b);
        }
    }
    out
}

fn preview<S: AsRef<str>>(s: S, max: usize) -> String {
    let s = s.as_ref();
    if s.chars().count() <= max {
        s.to_string()
    } else {
        let mut out = String::new();
        for (i, ch) in s.chars().enumerate() {
            if i >= max { break; }
            out.push(ch);
        }
        out.push('…');
        out
    }
}

fn sample_char(rng: &mut StdRng) -> char {
    let bucket: u32 = rng.gen_range(0..14);
    match bucket {
        0..=5 => *INTERESTING_CHARS.choose(rng).unwrap(),
        6..=8 => *ASCII_WHITESPACE.choose(rng).unwrap(),
        9..=10 => *ASCII_ALNUM.choose(rng).unwrap(),
        11..=12 => *ASCII_PUNCT.choose(rng).unwrap(),
        _ => random_unicode(rng),
    }
}

fn random_unicode(rng: &mut StdRng) -> char {
    loop {
        let value: u32 = rng.gen_range(0..=0x10FFFF);
        if let Some(ch) = char::from_u32(value) {
            return ch;
        }
    }
}

fn collect_fancy_spans(text: &str) -> Vec<(usize, usize)> {
    use fancy_regex::Regex;
    use std::sync::OnceLock;

    static INSTANCE: OnceLock<Regex> = OnceLock::new();
    let regex = INSTANCE.get_or_init(|| Regex::new(tiktoken::cl100k::CL100K_PATTERN).unwrap());
    regex
        .find_iter(text)
        .map(|res| {
            let m = res.expect("fancy-regex error while tokenizing");
            (m.start(), m.end())
        })
        .collect()
}

fn collect_custom_spans(text: &str, parser: &Cl100kParser) -> Vec<CustomSpan> {
    parser
        .find_iter(text)
        .map(|m| CustomSpan {
            start: m.start(),
            end: m.end(),
            kind: m.kind(),
        })
        .collect()
}

fn compare_spans(
    text: &str,
    fancy: &[(usize, usize)],
    custom: &[CustomSpan],
) -> Result<(), String> {
    let custom_pairs: Vec<_> = custom.iter().map(|span| (span.start, span.end)).collect();
    if custom_pairs == fancy {
        return Ok(());
    }

    if fancy.len() != custom.len() {
        let matching_prefix = fancy
            .iter()
            .zip(custom_pairs.iter())
            .take_while(|(a, b)| a == b)
            .count();
        return Err(format!(
            "Span count mismatch: fancy={} custom={} (matching prefix spans={})",
            fancy.len(),
            custom.len(),
            matching_prefix
        ));
    }

    for (idx, ((f_start, f_end), span)) in fancy.iter().zip(custom.iter()).enumerate() {
        if *f_start != span.start || *f_end != span.end {
            let fancy_slice = &text[*f_start..*f_end];
            let custom_slice = &text[span.start..span.end];
            let fancy_snippet = escape_snippet(fancy_slice, 32);
            let custom_snippet = escape_snippet(custom_slice, 32);
            return Err(format!(
                "Mismatch at span {idx}: fancy [{f_start},{f_end}) \"{fancy_snippet}\" vs custom [{c_start},{c_end}) \"{custom_snippet}\" ({:?})",
                span.kind,
                c_start = span.start,
                c_end = span.end
            ));
        }
    }

    Err("Spans differ but mismatch could not be isolated".to_string())
}

fn escape_snippet(slice: &str, limit: usize) -> String {
    if slice.is_empty() {
        return "<empty>".to_string();
    }
    let mut out = String::new();
    let mut count = 0;
    for ch in slice.chars() {
        if count >= limit {
            out.push('…');
            break;
        }
        out.extend(ch.escape_default());
        count += 1;
    }
    out
}

fn dump_bytes(text: &str) {
    let bytes = text.as_bytes();
    const PER_ROW: usize = 16;
    for (row, chunk) in bytes.chunks(PER_ROW).enumerate() {
        print!("{:04X}:", row * PER_ROW);
        for byte in chunk {
            print!(" {:02X}", byte);
        }
        println!();
    }
}

fn dump_tokens(tokens: &[Rank]) {
    const PER_ROW: usize = 32;
    for (row, chunk) in tokens.chunks(PER_ROW).enumerate() {
        print!("{:04X}:", row * PER_ROW);
        for t in chunk {
            print!(" {}", t);
        }
        println!();
    }
}

trait ChooseExt<T> {
    fn choose<'a>(&'a self, rng: &mut StdRng) -> Option<&'a T>;
}

impl<T> ChooseExt<T> for [T] {
    fn choose<'a>(&'a self, rng: &mut StdRng) -> Option<&'a T> {
        if self.is_empty() {
            None
        } else {
            let idx = rng.gen_range(0..self.len());
            Some(&self[idx])
        }
    }
}
