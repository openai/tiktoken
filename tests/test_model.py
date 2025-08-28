import tiktoken
def test_gpt5_encoding():
    encoding = tiktoken.encoding_for_model("gpt-5")
    assert encoding.name == "o200k_base"
    tokens = encoding.encode("hello world")
    assert tokens == [24912, 2375]  # Verify o200k_base behavior