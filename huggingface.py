import os
import json
import shutil
from transformers import AutoTokenizer

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Huggingface tokenizer to tiktoken tokenizer")
    parser.add_argument("--hf_name", type=str, default="Upstage/SOLAR-10.7B-Instruct-v1.0", help="huggingface model name")
    parser.add_argument("--save_fn", type=str, default="/Users/junhyunpark/playground/just_pr/tiktoken/vocab/tiktoken_solar_en2.json", help="save_filename")

    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained("Upstage/SOLAR-10.7B-Instruct-v1.0")
    if not os.path.exists("./.tmp"):
        os.makedirs("./.tmp") 
    tokenizer.save_pretrained("./.tmp")

    with open('./.tmp/tokenizer.json') as f:
        vocab = json.load(f)

    new_vocab = {}
    for k, v in vocab['model']['vocab'].items():
        if k not in ['<s>', '</s>', '<unk>']:
            new_vocab[k] = v

    with open(args.save_fn, 'w') as f:
        json.dump(new_vocab, f, indent=2, ensure_ascii=False)

    shutil.rmtree("./.tmp")

if __name__ == "__main__":
    main()