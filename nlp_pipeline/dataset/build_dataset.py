import os
from dataclasses import dataclass, field
from typing import Optional

import datasets
import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    BartTokenizer,
    HfArgumentParser,
    T5Tokenizer,
)

from constant import SCRIPT_PATH
from nlp_pipeline.dataset.build_transformer_format_dataset.processor import DataProcessor
from nlp_pipeline.utils.file_utils import logger


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    main_task: str = field(
        default="question-generation",
        metadata={"help": "Which task 'question-generation', 'question-answering'"},
    )

    training_task: str = field(
        default="multitask",
        metadata={"help": "One of 'qa', 'qg', 'answer extraction'"},
    )
    output_dir: str = field(
        default=None,
        metadata={
            "help": "The output directory where the processed data will be saved."
        },
    )
    model_type: str = field(default=None, metadata={"help": "One of 't5', 'bart'"})
    model_name_or_path: str = field(
        default=None,
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        },
    )
    pretrained_tokenizer_name_or_path: str = field(
        default=None,
        metadata={
            "help": "Path to pretrained tokenizer or model name in huggingface.co/models"
        },
    )
    customized_tokenizer_save_path: str = field(
        default=None, metadata={"help": "Path to save customized tokenizer"}
    )
    dataset_train_path: Optional[str] = field(
        default="data/dummy_data", metadata={"help": "Path for train dataset directory"}
    )
    dataset_valid_path: Optional[str] = field(
        default="data/dummy_data", metadata={"help": "Path for valid dataset directory"}
    )
    dataset_test_path: Optional[str] = field(
        default="data/dummy_data", metadata={"help": "Path for test dataset directory"}
    )
    train_file_name: Optional[str] = field(
        default=None, metadata={"help": "Name for cached train dataset"}
    )
    valid_file_name: Optional[str] = field(
        default=None, metadata={"help": "Name for cached valid dataset"}
    )
    test_file_name: Optional[str] = field(
        default=None, metadata={"help": "Name for cached test dataset"}
    )
    valid_for_qg_only: bool = field(
        default=False,
        metadata={
            "help": "For multitask dataset valid split should contain only qg task or all tasks."
        },
    )
    qg_format: Optional[str] = field(
        default="highlight_qg_format",
        metadata={
            "help": "How to format inputs for question generation, 'highlight_qg_format' or 'prepend_qg_format'"
        },
    )
    max_source_length: Optional[int] = field(
        default=512, metadata={"help": "Max input length for the source text"}
    )
    max_target_length: Optional[int] = field(
        default=32, metadata={"help": "Max input length for the target text"}
    )


def filter_qg(example):
    return example["task"] == "qg"


def filter_qa(example):
    return example["task"] == "qa"


TRAINING_TASK = {
    "multitask": ("qg", "qa", "answer_ext"),
    "qg": ("qg",),
    "qa": ("qa",)
}

def build_dataset_from_script(
    main_task: str = "question-generation",
    training_task: str = "multitask",
    qg_format: str = "highlight_qg_format",
    dataset_train_path: str = None,
    dataset_valid_path: str = None,
    dataset_test_path: str = None,
    output_dir: str = None,
    pretrained_tokenizer_name_or_path: str = "t5-base",
    customized_tokenizer_save_path: str = "t5_qg_tokenizer",
    model_type: str = "t5",
    max_source_length: int = 512,
    max_target_length: int = 64,
    train_file_name: str = "train_data_hl_t5.pt",
    valid_file_name: str = "valid_data_hl_t5.pt",
    test_file_name: str = "test_data_hl_t5.pt",
):
    if model_type == "t5":
        tokenizer = T5Tokenizer.from_pretrained(pretrained_tokenizer_name_or_path)
    else:
        tokenizer = BartTokenizer.from_pretrained(pretrained_tokenizer_name_or_path)

    tokenizer.add_tokens(["<sep>", "<hl>"])
    name_dataset = ["train", "validation", "test"]
    dataset_train_abs_path = os.path.abspath(dataset_train_path)
    dataset_valid_abs_path = os.path.abspath(dataset_valid_path)
    dataset_test_abs_path = os.path.abspath(dataset_test_path)

    dataset = datasets.load_dataset(
        SCRIPT_PATH,
        ignore_verifications=True,
        name=qg_format,
        data_files={
            "train": dataset_train_abs_path,
            "validation": dataset_valid_abs_path,
            "test": dataset_test_abs_path,
        },
        sub_task=TRAINING_TASK[training_task],
    )

    train_dataset, valid_dataset, test_dataset = [
        dataset[name] for name in name_dataset
    ]
    lm_processor = DataProcessor(
        tokenizer,
        model_type=model_type,
        max_source_length=max_source_length,
        max_target_length=max_target_length,
    )

    if main_task == "question-generation":
        logger.info("Processing valid and test data only for qg task")
        valid_dataset = valid_dataset.filter(filter_qg)
        test_dataset = test_dataset.filter(filter_qg)

    elif main_task == "question-answering":
        logger.info("Processing valid and test data only for qa task")
        valid_dataset = valid_dataset.filter(filter_qa)
        test_dataset = test_dataset.filter(filter_qa)

    # Process dataset for language model
    train_dataset = lm_processor.process(train_dataset)
    valid_dataset = lm_processor.process(valid_dataset)
    test_dataset = lm_processor.process(test_dataset)

    columns = ["source_ids", "target_ids", "attention_mask"]
    train_dataset.set_format(type="torch", columns=columns)
    valid_dataset.set_format(type="torch", columns=columns)
    test_dataset.set_format(type="torch", columns=columns)

    if not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir, exist_ok=True)
            logger.info(f"Directory '{output_dir}' created successfully")
        except OSError as error:
            logger.info(f"Directory '{output_dir}' can not be created, {error}")

    if train_file_name is None:
        train_file_name = f"train_data_{qg_format}_{model_type}.pt"
        train_path = os.path.join(output_dir, train_file_name)

        valid_file_name = f"valid_data_{qg_format}_{model_type}.pt"
        valid_path = os.path.join(output_dir, valid_file_name)

        test_file_name = f"test_data_{qg_format}_{model_type}.pt"
        test_path = os.path.join(output_dir, test_file_name)
    else:
        train_path = os.path.join(output_dir, train_file_name)
        valid_path = os.path.join(output_dir, valid_file_name)
        test_path = os.path.join(output_dir, test_file_name)

    torch.save(train_dataset, train_path)
    logger.info(f"Saved train dataset at {train_path}")

    torch.save(valid_dataset, valid_path)
    logger.info(f"Saved validation dataset at {valid_path}")

    torch.save(test_dataset, test_path)
    logger.info(f"Saved test dataset at {test_path}")

    if not os.path.exists(customized_tokenizer_save_path):
        os.makedirs(customized_tokenizer_save_path, exist_ok=True)

    tokenizer.save_pretrained(customized_tokenizer_save_path)
    logger.info(f"Saved tokenizer at {customized_tokenizer_save_path}")


def main():
    parser = HfArgumentParser((DataTrainingArguments,))
    data_args = parser.parse_args_into_dataclasses()[0]

    if data_args.model_type == "t5":
        tokenizer = T5Tokenizer.from_pretrained(
            data_args.pretrained_tokenizer_name_or_path
        )
    else:
        tokenizer = BartTokenizer.from_pretrained(
            data_args.pretrained_tokenizer_name_or_path
        )

    tokenizer.add_tokens(["<sep>", "<hl>"])

    build_dataset_from_script(
        main_task=data_args.main_task,
        model_type=data_args.model_type,
        dataset_train_path=data_args.dataset_train_path,
        dataset_valid_path=data_args.dataset_valid_path,
        dataset_test_path=data_args.dataset_test_path,
        pretrained_tokenizer_name_or_path=data_args.pretrained_tokenizer_name_or_path,
        customized_tokenizer_save_path=data_args.customized_tokenizer_save_path,
        output_dir=data_args.output_dir,
        train_file_name=data_args.train_file_name,
        valid_file_name=data_args.valid_file_name,
        test_file_name=data_args.test_file_name,
    )


if __name__ == "__main__":
    main()
