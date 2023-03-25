from __future__ import annotations, print_function

import argparse

from . import list_encoding_names, get_encoding, encoding_for_model
from .model import MODEL_TO_ENCODING


DEFAULT_ENCODING = "cl100k_base"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m tiktoken.tool",
        description="tiktoken is a fast BPE tokeniser for use with OpenAI's models"
    )

    parser.add_argument(
        "file",
        type=argparse.FileType("r"),
        help="input file",
    )

    encoding_group = parser.add_mutually_exclusive_group()
    encoding_group.add_argument(
        "-e",
        "--encoding",
        type=str,
        choices=list_encoding_names(),
        metavar="ENCODING",
        default=DEFAULT_ENCODING,
        help="encoding to use",
    )
    encoding_group.add_argument(
        "-m",
        "--model",
        type=str,
        choices=MODEL_TO_ENCODING.keys(),
        metavar="MODEL",
        help="model to use to determine encoding",
    )

    parser.add_argument(
        "-d",
        "--decode",
        action="store_true",
        help="decode/detokenize file containing one token per line",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.model:
        encoding = encoding_for_model(args.model)
    else:
        encoding = get_encoding(args.encoding)

    if args.decode:
        lines = args.file.read().strip().splitlines()
        tokens = list(map(int, lines))
        print(encoding.decode(tokens), end="")
    else:
        tokens = encoding.encode(args.file.read())
        lines = "\n".join(map(str, tokens))
        print(lines)


if __name__ == "__main__":
    main()
