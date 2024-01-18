import json

def solar():
    with open("./vocab/tiktoken_solar_en.json") as f:
        raw_json = json.load(f)
    mergeable_ranks = {k.replace("▁", ' ').encode(): v for k,v in raw_json.items()}

    return {
        "name": "solar",
        "explicit_n_vocab": 32000,
        "pat_str": r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""",
        "mergeable_ranks": mergeable_ranks,
        "special_tokens": {"<unk>": 0,
                           "<s>": 1,
                           "</s>": 2},
    }

ENCODING_CONSTRUCTORS = {
    "solar": solar,
}
