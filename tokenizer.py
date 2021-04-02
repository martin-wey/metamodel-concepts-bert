"""
Train a byte-pair encoding tokenizer using HuggingFace's implementation
see: https://huggingface.co/transformers/main_classes/tokenizer.html
"""

import os
import argparse

from tokenizers.implementations import ByteLevelBPETokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--train_file', type=str,
        help='The input training data file (a text file).'
    )
    parser.add_argument(
        '--output_dir', type=str,
        help='Path where the tokenizer should be saved.'
    )
    parser.add_argument(
        '--max_seq_length', type=int, default=512,
        help='Maximum length of the sequences considered during BPE.'
    )
    parser.add_argument(
        '--vocab_size', type=int, default=10000,
        help='Size of the vocabulary (=number of merges)'
    )
    parser.add_argument(
        '--min_frequency', type=int, default=2,
        help='Minimum frequency of a token. By default ignore hapax legomena.'
    )
    args = parser.parse_args()

    tokenizer = ByteLevelBPETokenizer()
    tokenizer.enable_truncation(args.max_seq_length)

    # These are the special tokens used by the RoBERTa model and should not be changed.
    #   see: https://huggingface.co/transformers/model_doc/roberta.html#robertatokenizer
    special_tokens = ['<s>', '<pad>', '</s>', '<unk>', '<mask>']
    tokenizer.train(
        files=[args.train_file],
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        show_progress=True,
        special_tokens=special_tokens
    )

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    tokenizer.save_model(args.output_dir)


if __name__ == '__main__':
    main()
