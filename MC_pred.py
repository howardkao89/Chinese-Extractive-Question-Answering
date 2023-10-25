import argparse
import json
import os
import numpy as np
from dataclasses import dataclass
from itertools import chain
from typing import Optional, Union

from accelerate.logging import get_logger
from datasets import load_dataset


import transformers
from transformers import (
    MODEL_MAPPING,
    AutoConfig,
    AutoModelForMultipleChoice,
    AutoTokenizer,
    PreTrainedTokenizerBase,
    Trainer
)
from transformers.utils import PaddingStrategy


logger = get_logger(__name__)
# You should update this to your particular problem to have better documentation of `model_type`
MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


def parse_args():
    parser = argparse.ArgumentParser(description="Test a transformers model on a multiple choice task")
    parser.add_argument(
        "--test_file", type=str, default=None, help="A csv or a json file containing the testing data."
    )
    parser.add_argument(
        "--context_file", type=str, default=None, help="A csv or a json file containing the context data."
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=128,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_lengh` is passed."
        ),
    )
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=False,
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument("--output_path", type=str, default=None, help="Where to store the final model.")
    parser.add_argument(
        "--trust_remote_code",
        type=bool,
        default=False,
        help=(
            "Whether or not to allow for custom models defined on the Hub in their own modeling files. This option"
            "should only be set to `True` for repositories you trust and in which you have read the code, as it will "
            "execute code present on the Hub on your local machine."
        ),
    )

    args = parser.parse_args()

    return args


@dataclass
class DataCollatorForMultipleChoice:

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = 512
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features):
        batch_size = len(features)
        num_choices = len(features[0]["input_ids"])
        flattened_features = [
            [{k: v[i] for k, v in feature.items()} for i in range(num_choices)] for feature in features
        ]
        flattened_features = list(chain(*flattened_features))

        batch = self.tokenizer.pad(
            flattened_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        batch = {k: v.view(batch_size, num_choices, -1) for k, v in batch.items()}
        return batch


def main():
    args = parse_args()

    data_files = {}
    if args.test_file is not None:
        data_files["test"] = args.test_file
    extension = args.test_file.split(".")[-1]
    raw_datasets = load_dataset(extension, data_files=data_files)

    with open(args.context_file, "r") as context_file:
        context = json.load(context_file)
        
    test_paragraphs = {0: [], 1: [], 2: [], 3: []}
    for dict in raw_datasets["test"]:
        for idx in range(4):
            test_paragraphs[idx].append(context[dict["paragraphs"][idx]])
    for idx in range(4):
        raw_datasets["test"] = raw_datasets["test"].add_column(f"paragraphs_{idx}", test_paragraphs[idx])

    ending_names = [f"paragraphs_{i}" for i in range(4)]
    context_name = "question"
    question_header_name = "id"

    config = AutoConfig.from_pretrained(args.model_name_or_path, trust_remote_code=args.trust_remote_code)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=not args.use_slow_tokenizer, trust_remote_code=args.trust_remote_code)
    model = AutoModelForMultipleChoice.from_pretrained(args.model_name_or_path, from_tf=bool(".ckpt" in args.model_name_or_path), config=config, trust_remote_code=args.trust_remote_code)

    padding = "max_length" if args.pad_to_max_length else False

    def preprocess_function(examples):
        first_sentences = [[context] * 4 for context in examples[context_name]]
        question_headers = examples[question_header_name]
        second_sentences = [
            [f"{header} {examples[end][i]}" for end in ending_names] for i, header in enumerate(question_headers)
        ]
        # Flatten out
        first_sentences = list(chain(*first_sentences))
        second_sentences = list(chain(*second_sentences))

        # Tokenize
        tokenized_examples = tokenizer(
            first_sentences,
            second_sentences,
            max_length=args.max_seq_length,
            padding=padding,
            truncation=True,
        )
        # Un-flatten
        tokenized_inputs = {k: [v[i : i + 4] for i in range(0, len(v), 4)] for k, v in tokenized_examples.items()}
        return tokenized_inputs

    processed_datasets = raw_datasets.map(preprocess_function, batched=True, remove_columns=raw_datasets["test"].column_names)

    test_dataset = processed_datasets["test"]

    trainer = Trainer(model=model, data_collator=DataCollatorForMultipleChoice(tokenizer), tokenizer=tokenizer)

    pred = np.argmax(trainer.predict(test_dataset).predictions, axis=1)

    with open(args.test_file, "r") as test_file:
        test = json.load(test_file)

    for idx in range(np.size(pred)):
        test[idx]["relevant"] = test[idx]["paragraphs"][pred[idx]]

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, "w", encoding='utf-8') as output_file:
        json.dump(test, output_file, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    main()
