import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, HfArgumentParser

from questgen.pipelines.pipeline import pipeline
from questgen.ranking.processor import DataProcessor
from questgen.utils.file_utils import load_json_file, logger, write_json_file
from questgen.utils.model_utils import extract_features


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        default="output/models/v1.2/t5-en-vi-base-multitask",
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        },
    )
    tokenizer_name_or_path: Optional[str] = field(
        default="output/models/v1.2/t5-en-vi-base-multitask",
        metadata={
            "help": "Pretrained tokenizer name or path if not the same as model_name"
        },
    )


@dataclass
class DataTrainingArguments:
    train_batch_size: Optional[int] = field(default=4)
    eval_batch_size: Optional[int] = field(default=4)
    test_batch_size: Optional[int] = field(default=1)

    train_data_path: Optional[str] = field(
        default="data/qa_ranking/json/train_data_v1.2.json",
        metadata={"help": "Path for cached train dataset"},
    )

    eval_data_path: Optional[str] = field(
        default="data/qa_ranking/json/dev_data_v1.2.json",
        metadata={"help": "Path for cached eval dataset"},
    )

    test_data_path: Optional[str] = field(
        default=None, metadata={"help": "Path for cached test dataset"}
    )

    max_length: Optional[int] = field(
        default=256, metadata={"help": "Max input length for the source text"}
    )

    save_features_qa: Optional[str] = field(
        default="data/qa_ranking/npy",
        metadata={"help": "Path to save features to train qa_ranking module"},
    )


def return_pairs(context):
    pair = pipeline(context)
    for idc in range(len(pair)):
        pair[idc]["context"] = context
    return pair


def create_pairs(data_path, save_path=None):
    test_data = load_json_file(data_path)
    context = list(map(lambda data: data["context"], test_data))
    # remove duplicate context
    context = list(dict.fromkeys(context))

    qa_pair = list(map(return_pairs, context))
    qa_pair = pd.DataFrame(qa_pair[0]).drop_duplicates()
    qa_pair = qa_pair.to_dict(orient="records")

    if save_path is not None:
        write_json_file(qa_pair, save_path)


def main(args_file=None, task="extract-features"):
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments))

    if (len(sys.argv) == 2 and sys.argv[1].endswith(".json")) or args_file is not None:
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        args_file_path = (
            os.path.abspath(sys.argv[1]) if args_file is None else args_file
        )
        model_args, data_args = parser.parse_json_file(json_file=args_file_path)
    else:
        model_args, data_args = parser.parse_args_into_dataclasses()

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_args.tokenizer_name_or_path)

    if task == "qa_pairs":
        create_pairs(
            data_path="dataset/viquad_qg/transformer_format/full_data/test_viquad.json"
        )

    elif task == "extract-features":
        train_data = load_json_file(data_args.train_data_path)
        eval_data = load_json_file(data_args.eval_data_path)

        assert [train_data[idc]["targets"] == 1 for idc in range(len(train_data))]
        logger.info(f"the number of train data %s {len(train_data)}")
        logger.info(f"the number of eval data %s {len(eval_data)}")

        processor = DataProcessor(tokenizer=tokenizer, max_length=data_args.max_length)

        train_loader = processor.process(
            train_data, bz=data_args.train_batch_size, mode="train"
        )
        feature_train = extract_features(
            dataloader=train_loader, model=model, verbose=False, device="cuda"
        )

        assert feature_train.shape[0] == len(
            train_data
        ), f"the shape of feature_train {feature_train.shape[0]} must be equal dev_data {len(train_data)}"

        np.save(
            Path(f"{data_args.save_features_qa}", "feature_train.npy"), feature_train
        )

        eval_loader = processor.process(
            eval_data, bz=data_args.eval_batch_size, mode="eval"
        )
        feature_eval = extract_features(
            dataloader=eval_loader, model=model, verbose=False, device="cuda"
        )

        target_eval = [eval_data[idc]["targets"] for idc in range(len(eval_data))]

        assert feature_eval.shape[0] == len(
            target_eval
        ), f"The shape of feature_eval {feature_eval.shape[0]} must be equal target_eval {len(target_eval)}"
        assert feature_eval.shape[0] == len(
            eval_data
        ), f"The shape of feature_eval {feature_eval.shape[0]} must be equal dev_data {len(eval_data)}"

        np.save(Path(f"{data_args.save_features_qa}", "feature_eval.npy"), feature_eval)

        if data_args.test_data_path is not None:
            test_data = load_json_file(data_args.test_data_path)
            logger.info(f"the number of test data %s {len(test_data)}")

            test_loader = processor.process(
                test_data, bz=data_args.test_batch_size, mode="test"
            )
            feature_test = extract_features(
                dataloader=test_loader, model=model, device="cuda"
            )

            assert (feature_test.shape[0]) == len(
                test_data
            ), f"the shape of feature_test {feature_test.shape[0]} must be equal test_data {len(test_data)}"

            np.save(data_args.save_qa_test, feature_test)


if __name__ == "__main__":
    main(task="extract-features")
