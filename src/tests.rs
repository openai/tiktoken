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

    let enc = EncodingFactory::cl100k_base().unwrap();
    for token in 0..10000 {
        assert_eq!(
            enc.encode_single_token_bytes(&enc.decode_single_token_bytes(token).unwrap())
                .unwrap(),
            token
        );
    }
}
