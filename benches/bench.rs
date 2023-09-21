use criterion::{black_box, criterion_group, criterion_main, Criterion};
use tiktoken::{EncodingFactory, SpecialTokenHandling, SpecialTokenAction};

fn cl100k_base_benchmark(c: &mut Criterion) {
    let x = EncodingFactory::cl100k_base().unwrap();
    let y = tiktoken_rs::cl100k_base().unwrap();
    let t = include_str!("test.txt");
    c.bench_function("cl100k_base_estimation", |b| {
        b.iter(|| {
            black_box(x.estimate_num_tokens_no_special_tokens_fast(
                &t,
            ));
        });
    });
    c.bench_function("cl100k_base", |b| {
        b.iter(|| {
            black_box(x.encode(
                &t,
                &SpecialTokenHandling {
                    default: SpecialTokenAction::Special,
                    ..Default::default()
                }
            )
            .unwrap());
        });
    });
    c.bench_function("cl100k_base_ordinary", |b| {
        b.iter(|| {
            black_box(x.encode_ordinary(
                &t,
            ));
        });
    });
    c.bench_function("cl100k_base_tiktoken-rs", |b| {
        b.iter(|| {
            black_box(y.encode_with_special_tokens(
                &t
            ));
        });
    });
    c.bench_function("cl100k_base_50atatime", |b| {
        b.iter(|| {
            black_box(t.chars()
                .collect::<Vec<char>>()
                .chunks(50)
                .map(|chunk| {
                    x.encode(
                        &chunk.iter().collect::<String>(),
                        &SpecialTokenHandling {
                            default: SpecialTokenAction::Special,
                            ..Default::default()
                        }
                    )
                    .unwrap().len()
                })
                .collect::<Vec<_>>());
        });
    });
    let y = x.encode(
        &t,
        &SpecialTokenHandling {
            default: SpecialTokenAction::Special,
            ..Default::default()
        }
    )
    .unwrap();
    println!("num tokens: {:?}", y.len());
}

criterion_group!(benches, cl100k_base_benchmark);
criterion_main!(benches);