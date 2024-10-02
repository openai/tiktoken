import tiktoken


def test_pickle():
    import pickle

    enc_old = tiktoken.get_encoding("r50k_base")
    enc_new = pickle.loads(pickle.dumps(enc_old))
    assert enc_old.encode("hello world") == enc_new.encode("hello world")

    enc_old = tiktoken.Encoding(
        name="custom_enc",
        pat_str=enc_old._pat_str,
        mergeable_ranks=enc_old._mergeable_ranks,
        special_tokens={"<|pickle|>": 100_000},
    )
    enc_new = pickle.loads(pickle.dumps(enc_old))
    assert enc_old.encode("hello world") == enc_new.encode("hello world")
    assert (
        enc_old.encode("<|pickle|>", allowed_special="all")
        == enc_new.encode("<|pickle|>", allowed_special="all")
        == [100_000]
    )
