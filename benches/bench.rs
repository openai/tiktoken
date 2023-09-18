use criterion::{black_box, criterion_group, criterion_main, Criterion};
use tiktoken::{EncodingFactory, SpecialTokenHandling, SpecialTokenAction};

fn cl100k_base_benchmark(c: &mut Criterion) {
    let x = EncodingFactory::cl100k_base().unwrap();
    let t = "This feature allows you to pause the execution of your JavaScript code whenever an exception occurs, without moving the focus away from the current context. You can then inspect the current state of your application at the moment the exception was thrown.
    ";
    // repeat t 200 times
    let t = std::iter::repeat(t).take(200).collect::<String>();
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