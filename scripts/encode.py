#!/usr/bin/env python3
import argparse
import tiktoken

def main():
    # Parse arguments
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "text",
        help="The input to tokenize",
    )
    parser.add_argument(
        "--model",
        help="Name of model for which tokens will be generated, changed which encoding is used",
        default="gpt-3.5-turbo-0613",
    )

    args = parser.parse_args()

    # Encode
    encoder = tiktoken.encoding_for_model(args.model)

    encoded = encoder.encode(args.text)

    print(f"Encoded '{args.text}' as '{encoded}'")

if __name__ == '__main__':
    main()