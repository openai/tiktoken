from tiktoken.load import data_gym_to_mergeable_bpe_ranks, load_tiktoken_bpe
import os

ENDOFTEXT = "<|endoftext|>"
FIM_PREFIX = "<|fim_prefix|>"
FIM_MIDDLE = "<|fim_middle|>"
FIM_SUFFIX = "<|fim_suffix|>"
ENDOFPROMPT = "<|endofprompt|>"


def gpt2():
    vocab_bpe_file = os.environ.get(
        "TIKTOKEN_BPE_FILE_GPT2_VOCAB",
        "https://openaipublic.blob.core.windows.net/gpt-2/encodings/main/vocab.bpe",
    )
    encoder_json_file = os.environ.get(
        "TIKTOKEN_BPE_FILE_GPT2_ENCODER",
        "https://openaipublic.blob.core.windows.net/gpt-2/encodings/main/encoder.json",
    )
    mergeable_ranks = data_gym_to_mergeable_bpe_ranks(
        vocab_bpe_file=vocab_bpe_file,
        encoder_json_file=encoder_json_file,
    )
    return {
        "name": "gpt2",
        "explicit_n_vocab": 50257,
        "pat_str": r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""",
        "mergeable_ranks": mergeable_ranks,
        "special_tokens": {ENDOFTEXT: 50256},
    }


def r50k_base():
    tiktoken_bpe_file = os.environ.get(
        "TIKTOKEN_BPE_FILE_R50K_BASE",
        "https://openaipublic.blob.core.windows.net/encodings/r50k_base.tiktoken",
    )
    mergeable_ranks = load_tiktoken_bpe(tiktoken_bpe_file)
    return {
        "name": "r50k_base",
        "explicit_n_vocab": 50257,
        "pat_str": r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""",
        "mergeable_ranks": mergeable_ranks,
        "special_tokens": {ENDOFTEXT: 50256},
    }


def p50k_base():
    tiktoken_bpe_file = os.environ.get(
        "TIKTOKEN_BPE_FILE_P50K_BASE",
        "https://openaipublic.blob.core.windows.net/encodings/p50k_base.tiktoken",
    )
    mergeable_ranks = load_tiktoken_bpe(tiktoken_bpe_file)
    return {
        "name": "p50k_base",
        "explicit_n_vocab": 50281,
        "pat_str": r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""",
        "mergeable_ranks": mergeable_ranks,
        "special_tokens": {ENDOFTEXT: 50256},
    }


def p50k_edit():
    tiktoken_bpe_file = os.environ.get(
        "TIKTOKEN_BPE_FILE_P50K_BASE",
        "https://openaipublic.blob.core.windows.net/encodings/p50k_base.tiktoken",
    )
    mergeable_ranks = load_tiktoken_bpe(tiktoken_bpe_file)
    special_tokens = {
        ENDOFTEXT: 50256,
        FIM_PREFIX: 50281,
        FIM_MIDDLE: 50282,
        FIM_SUFFIX: 50283,
    }
    return {
        "name": "p50k_edit",
        "pat_str": r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""",
        "mergeable_ranks": mergeable_ranks,
        "special_tokens": special_tokens,
    }


def cl100k_base():
    tiktoken_bpe_file = os.environ.get(
        "TIKTOKEN_BPE_FILE_CL100K_BASE",
        "https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken",
    )
    mergeable_ranks = load_tiktoken_bpe(tiktoken_bpe_file)
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
