use crate::{
    encoding::SpecialTokenAction, encoding::SpecialTokenHandling, openai_public::EncodingFactory,
};

#[test]
fn test_simple() {
    let enc = EncodingFactory::cl100k_base().unwrap();
    assert_eq!(
        enc.encode(
            "hello world",
            &SpecialTokenHandling {
                default: SpecialTokenAction::Forbidden,
                ..Default::default()
            }
        )
        .unwrap(),
        vec![15339, 1917]
    );
    assert_eq!(enc.decode(&[15339, 1917]), "hello world");
    assert_eq!(
        enc.encode(
            "hello <|endoftext|>",
            &SpecialTokenHandling {
                default: SpecialTokenAction::Special,
                ..Default::default()
            }
        )
        .unwrap(),
        vec![15339, 220, 100257]
    );
    assert_eq!(
        enc.encode(
            "hello <|endoftext|>",
            &SpecialTokenHandling {
                default: SpecialTokenAction::Forbidden,
                overrides: vec![("<|endoftext|>".to_string(), SpecialTokenAction::Special)],
            }
        )
        .unwrap(),
        vec![15339, 220, 100257]
    );
    assert_eq!(
        enc.encode(
            "hello <|endoftext|>",
            &SpecialTokenHandling {
                default: SpecialTokenAction::NormalText,
                ..Default::default()
            }
        )
        .unwrap(),
        vec![15339, 83739, 8862, 728, 428, 91, 29]
    );
    assert_eq!(
        enc.encode(
            include_str!("test.txt"),
            &SpecialTokenHandling {
                default: SpecialTokenAction::NormalText,
                ..Default::default()
            }
        )
        .unwrap()
        .len(),
        7182 // this is same as text-davinici-003
    );
    assert_eq!(
        enc.encode(
            include_str!("prompt.txt"),
            &SpecialTokenHandling {
                default: SpecialTokenAction::NormalText,
                ..Default::default()
            }
        )
        .unwrap()
        .len(),
        6791 // this is same as text-davinici-003
    );

    let enc_r = EncodingFactory::r50k_base().unwrap();
    assert_eq!(
        enc_r
            .encode(
                "hello world    hello",
                &SpecialTokenHandling {
                    default: SpecialTokenAction::NormalText,
                    ..Default::default()
                }
            )
            .unwrap(),
        vec![31373, 995, 220, 220, 220, 23748] // this is the GPT-3 tokenizer
    );

    let enc_p = EncodingFactory::p50k_base().unwrap();
    assert_eq!(
        enc_p
            .encode(
                "hello world    hello",
                &SpecialTokenHandling {
                    default: SpecialTokenAction::NormalText,
                    ..Default::default()
                }
            )
            .unwrap(),
        vec![31373, 995, 50258, 23748] // this is the Codex tokenizer
    );

    assert_eq!(
        enc_p
            .encode(
                include_str!("prompt.txt"),
                &SpecialTokenHandling {
                    default: SpecialTokenAction::NormalText,
                    ..Default::default()
                }
            )
            .unwrap()
            .len(),
        9545 // this is same as text-davinici-003. HENCE TEXT-DAVINCI-003 USES CODEX TOKENIZER
    );

    let enc = EncodingFactory::cl100k_base().unwrap();
    for token in 0..10000 {
        assert_eq!(
            enc.encode_single_token_bytes(&enc.decode_single_token_bytes(token).unwrap())
                .unwrap(),
            token
        );
    }
}


#[test]
fn estimation_is_close() {
    let enc = EncodingFactory::cl100k_base().unwrap();

    let big = include_str!("big.txt");
    let test = include_str!("test.txt");
    let test2 = include_str!("test2.txt");


    let files = [&big, &test, &test2];
    for file in files.iter() {
        let real_count =
            enc
                .encode(
                    file,
                    &SpecialTokenHandling {
                        default: SpecialTokenAction::NormalText,
                        ..Default::default()
                    }
                )
                .unwrap()
                .len();

        let estimated_count = enc.estimate_num_tokens_no_special_tokens_fast(file);

        println!("Real count: {}", real_count);
        println!("Estimated count: {}", estimated_count);

        assert!((real_count as f64 - estimated_count as f64).abs() < 0.05 * real_count as f64);
    }
}

#[test]
fn simple_estimation_is_close() {
    let enc = EncodingFactory::cl100k_base().unwrap();

    let test = include_str!("tiny.txt");

    {
        let real_count =
            enc
                .encode(
                    &test,
                    &SpecialTokenHandling {
                        default: SpecialTokenAction::NormalText,
                        ..Default::default()
                    }
                )
                .unwrap()
                .len();

        let estimated_count = enc.estimate_num_tokens_no_special_tokens_fast(&test);

        println!("Real count: {}", real_count);
        println!("Estimated count: {}", estimated_count);

        assert!((real_count as f64 - estimated_count as f64).abs() < 0.05 * real_count as f64);
    }
}
