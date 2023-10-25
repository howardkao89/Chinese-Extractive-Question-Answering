import argparse
import json
import os
import csv

from accelerate.logging import get_logger
from datasets import load_dataset
from utils_qa import postprocess_qa_predictions

import transformers
from transformers import (
    MODEL_MAPPING,
    AutoConfig,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer
)
from transformers.utils.versions import require_version


require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/question-answering/requirements.txt")

logger = get_logger(__name__)
# You should update this to your particular problem to have better documentation of `model_type`
MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a Question Answering task")
    parser.add_argument(
        "--context_file", type=str, default=None, help="A csv or a json file containing the context data."
    )
    parser.add_argument(
        "--test_file", type=str, default=None, help="A csv or a json file containing the Prediction data."
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=384,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_lengh` is passed."
        ),
    )
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_seq_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=False,
    )
    parser.add_argument("--output_path", type=str, default=None, help="Where to store the final model.")
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )
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

    # Sanity checks
    if args.test_file is None:
        raise ValueError("Need test file.")
    else:
        if args.test_file is not None:
            extension = args.test_file.split(".")[-1]
            assert extension in ["csv", "json"], "`test_file` should be a csv or a json file."

    return args


def main():
    args = parse_args()

    data_files = {}
    if args.test_file is not None:
        data_files["test"] = args.test_file
    extension = args.test_file.split(".")[-1]
    raw_datasets = load_dataset(extension, data_files=data_files)
    
    with open(args.context_file, "r") as context_file:
        context = json.load(context_file)
        
    test_context = []
    for dict in raw_datasets["test"]:
        for idx in range(4):
            if dict["paragraphs"][idx] == dict["relevant"]:
                test_context.append(context[dict["relevant"]])
    raw_datasets["test"] = raw_datasets["test"].add_column("context", test_context)

    config = AutoConfig.from_pretrained(args.model_name_or_path, trust_remote_code=args.trust_remote_code)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True, trust_remote_code=args.trust_remote_code)
    model = AutoModelForQuestionAnswering.from_pretrained(args.model_name_or_path, from_tf=bool(".ckpt" in args.model_name_or_path), config=config, trust_remote_code=args.trust_remote_code,)

    column_names = raw_datasets["test"].column_names

    question_column_name = "question" if "question" in column_names else column_names[0]
    context_column_name = "context" if "context" in column_names else column_names[1]

    pad_on_right = tokenizer.padding_side == "right"

    if args.max_seq_length > tokenizer.model_max_length:
        logger.warning(
            f"The max_seq_length passed ({args.max_seq_length}) is larger than the maximum length for the "
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )

    max_seq_length = min(args.max_seq_length, tokenizer.model_max_length)

    def prepare_validation_features(examples):
        examples[question_column_name] = [q.lstrip() for q in examples[question_column_name]]

        tokenized_examples = tokenizer(
            examples[question_column_name if pad_on_right else context_column_name],
            examples[context_column_name if pad_on_right else question_column_name],
            truncation="only_second" if pad_on_right else "only_first",
            max_length=max_seq_length,
            stride=128,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length" if args.pad_to_max_length else False,
        )

        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

        tokenized_examples["example_id"] = []

        for i in range(len(tokenized_examples["input_ids"])):
            sequence_ids = tokenized_examples.sequence_ids(i)
            context_index = 1 if pad_on_right else 0

            sample_index = sample_mapping[i]
            tokenized_examples["example_id"].append(examples["id"][sample_index])

            tokenized_examples["offset_mapping"][i] = [
                (o if sequence_ids[k] == context_index else None)
                for k, o in enumerate(tokenized_examples["offset_mapping"][i])
            ]

        return tokenized_examples

    predict_examples = raw_datasets["test"]
    predict_dataset = predict_examples.map(prepare_validation_features, batched=True, num_proc=1, remove_columns=column_names, load_from_cache_file=not args.overwrite_cache, desc="Running tokenizer on prediction dataset")

    data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=None)

    trainer = Trainer(model=model, data_collator=data_collator, tokenizer=tokenizer)

    raw_pred = trainer.predict(predict_dataset).predictions

    raw_pred = postprocess_qa_predictions(predict_examples, predict_dataset, raw_pred)

    pred = [[u, v] for u, v in raw_pred.items()]

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, mode="w", newline="", encoding="utf-8") as output_file:
        csv_writer = csv.writer(output_file)

        csv_writer.writerow(["id", "answer"])
        for row in pred:
            csv_writer.writerow(row)

if __name__ == "__main__":
    main()
