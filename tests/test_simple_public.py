import tiktoken


def test_simple():
    # Note that there are more actual tests, they're just not currently public :-)
    enc = tiktoken.get_encoding("gpt2")
    assert enc.encode("hello world") == [31373, 995]
    assert enc.decode([31373, 995]) == "hello world"
    assert enc.encode("hello <|endoftext|>", allowed_special="all") == [31373, 220, 50256]

    enc = tiktoken.get_encoding("cl100k_base")
    assert enc.encode("hello world") == [15339, 1917]
    assert enc.decode([15339, 1917]) == "hello world"
    assert enc.encode("hello <|endoftext|>", allowed_special="all") == [15339, 220, 100257]

    for enc_name in tiktoken.list_encoding_names():
        enc = tiktoken.get_encoding(enc_name)
        for token in range(10_000):
            assert enc.encode_single_token(enc.decode_single_token_bytes(token)) == token
