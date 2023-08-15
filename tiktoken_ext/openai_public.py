from tiktoken.load import data_gym_to_mergeable_bpe_ranks, load_tiktoken_bpe

ENDOFTEXT = "<|endoftext|>"
FIM_PREFIX = "<|fim_prefix|>"
FIM_MIDDLE = "<|fim_middle|>"
FIM_SUFFIX = "<|fim_suffix|>"
ENDOFPROMPT = "<|endofprompt|>"


def gpt2(local_vocab_bpe_path: str = None, local_encoding_path: str = None):
    if local_vocab_bpe_path:
        vocab_bpe_file = local_vocab_bpe_path
    else:
        vocab_bpe_file = "https://openaipublic.blob.core.windows.net/gpt-2/encodings/main/vocab.bpe"
    if local_encoding_path:
        encoder_json_file = local_encoding_path
    else:
        encoder_json_file = "https://openaipublic.blob.core.windows.net/gpt-2/encodings/main/encoder.json"
    mergeable_ranks = data_gym_to_mergeable_bpe_ranks(
        vocab_bpe_file=vocab_bpe_file,
        encoder_json_file=encoder_json_file,
    )
    return {
        "name": "gpt2",
        "explicit_n_vocab": 50257,
        "pat_str": r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""",
        "mergeable_ranks": mergeable_ranks,
        "special_tokens": {"<|endoftext|>": 50256},
    }


def r50k_base():
    mergeable_ranks = load_tiktoken_bpe(
        "https://openaipublic.blob.core.windows.net/encodings/r50k_base.tiktoken"
    )
    return {
        "name": "r50k_base",
        "explicit_n_vocab": 50257,
        "pat_str": r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""",
        "mergeable_ranks": mergeable_ranks,
        "special_tokens": {ENDOFTEXT: 50256},
    }


def p50k_base():
    mergeable_ranks = load_tiktoken_bpe(
        "https://openaipublic.blob.core.windows.net/encodings/p50k_base.tiktoken"
    )
    return {
        "name": "p50k_base",
        "explicit_n_vocab": 50281,
        "pat_str": r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""",
        "mergeable_ranks": mergeable_ranks,
        "special_tokens": {ENDOFTEXT: 50256},
    }


def p50k_edit():
    mergeable_ranks = load_tiktoken_bpe(
        "https://openaipublic.blob.core.windows.net/encodings/p50k_base.tiktoken"
    )
    special_tokens = {ENDOFTEXT: 50256, FIM_PREFIX: 50281, FIM_MIDDLE: 50282, FIM_SUFFIX: 50283}
    return {
        "name": "p50k_edit",
        "pat_str": r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""",
        "mergeable_ranks": mergeable_ranks,
        "special_tokens": special_tokens,
    }


def cl100k_base(local_encoding_path: str = None):
    if local_encoding_path:
        bpe_file = local_encoding_path
    else:
        bpe_file = "https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken"
    mergeable_ranks = load_tiktoken_bpe(bpe_file)
    special_tokens = {
        ENDOFTEXT: 100257,
        FIM_PREFIX: 100258,
        FIM_MIDDLE: 100259,
        FIM_SUFFIX: 100260,
        ENDOFPROMPT: 100276,
    }
    return {
        "name": "cl100k_base",
        "pat_str": r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+""",
        "mergeable_ranks": mergeable_ranks,
        "special_tokens": special_tokens,
    }


ENCODING_CONSTRUCTORS = {
    "gpt2": gpt2,
    "r50k_base": r50k_base,
    "p50k_base": p50k_base,
    "p50k_edit": p50k_edit,
    "cl100k_base": cl100k_base,
}
