import json

from tiktoken.load import data_gym_to_mergeable_bpe_ranks


def test_data_gym_vocab_without_trailing_newline(tmp_path):
    rank_to_intbyte = [b for b in range(2**8) if chr(b).isprintable() and chr(b) != " "]
    data_gym_byte_to_byte = {chr(b): b for b in rank_to_intbyte}
    n = 0
    for b in range(2**8):
        if b not in rank_to_intbyte:
            data_gym_byte_to_byte[chr(2**8 + n)] = b
            rank_to_intbyte.append(b)
            n += 1

    byte_to_data_gym = {b: char for char, b in data_gym_byte_to_byte.items()}
    encoder = {byte_to_data_gym[b]: rank for rank, b in enumerate(rank_to_intbyte)}
    encoder["ab"] = len(encoder)

    vocab_bpe_file = tmp_path / "vocab.bpe"
    vocab_bpe_file.write_text("#version: 0.2\na b", encoding="utf-8")
    encoder_json_file = tmp_path / "encoder.json"
    encoder_json_file.write_text(json.dumps(encoder), encoding="utf-8")

    ranks = data_gym_to_mergeable_bpe_ranks(str(vocab_bpe_file), str(encoder_json_file))

    assert ranks[b"ab"] == 256
