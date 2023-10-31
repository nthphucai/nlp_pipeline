import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional, Union

import torch
import wandb
from torch.utils.data import ConcatDataset
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    BartTokenizer,
    HfArgumentParser,
    T5Tokenizer,
    TrainingArguments,
    set_seed,
)
from dacite import from_dict

from nlp_pipeline.dataset.build_transformer_format_dataset.data_collator import (
    Text2TextDataCollator,
)
from nlp_pipeline.models.T5CopyGenerator import T5CopyGenerator
from nlp_pipeline.trainer.modules.mixed_finetune import BestRatioMixedFineTune
from nlp_pipeline.trainer.trainer_transformers import BaseTrainer
from nlp_pipeline.utils.file_utils import load_json_file, logger, read_yaml_file
from nlp_pipeline.utils.model_utils import assert_not_all_frozen, freeze_embeds
from nlp_pipeline.utils.utils import set_gpu


MODEL_TYPE_TO_TOKENIZER = {
    "t5": T5Tokenizer,
    "bart": BartTokenizer,
    "t5-copy-enhance": T5Tokenizer,
}
TRAINING_ARGS_MAP = {
    "onnx": "ORTSeq2SeqTrainingArguments",
    "default": TrainingArguments,
}

TRAINER_ARGS_MAP = {"onnx": "ORTSeq2SeqTrainer", "default": BaseTrainer}


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        }
    )
    model_type: str = field(metadata={"help": "One of 't5', 't5-copy-enhance', 'bart'"})
    model_config: Optional[str] = field(
        default=None, metadata={"help": "Hyperparameter config for model"}
    )
    tokenizer_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained tokenizer name or path if not the same as model_name"
        },
    )
    onnx_mode: bool = field(
        default=False, metadata={"help": "Support training with ONNX and Optimum."}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where do you want to store the pretrained models downloaded from s3"
        },
    )
    label_smoothing: Optional[float] = field(
        default=0,
        metadata={
            "help": "Label smoothing rate, set to > 0 if you want to enable lable smoothing"
        },
    )
    freeze_embeds: bool = field(
        default=False,
        metadata={
            "help": "Freeze token embeddings and positional embeddings for BART, just token embeddings for T5."
        },
    )

    mix_finetune: bool = field(
        default=True, metadata={"help": "Whether to use mix finetune."}
    )

    project_name: str = field(
        default="question-generation",
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    train_file_path: str = field(metadata={"help": "Path for cached train dataset"})
    valid_file_path: str = field(metadata={"help": "Path for cached valid dataset"})
    train_file_path_new: str = field(
        default=None, metadata={"help": "Path for new cached train dataset"}
    )
    valid_file_path_new: str = field(
        default=None, metadata={"help": "Path for new cached valid dataset"}
    )
    data_dir: Optional[str] = field(
        default=None, metadata={"help": "Path for data files"}
    )
    qg_format: Optional[str] = field(
        default="prepend_qg_format",
        metadata={
            "help": "How to format inputs for que generation, 'highlight_qg_format' or 'prepend_qg_format'"
        },
    )
    max_source_length: Optional[int] = field(
        default=512, metadata={"help": "Max input length for the source text"}
    )
    max_target_length: Optional[int] = field(
        default=128, metadata={"help": "Max input length for the target text"}
    )


class QuestGenTrainer:
    def __init__(
        self,
        config_path: Optional[str] = None,
        model_name_or_path: Optional[str] = None,
        train_file_path: Optional[str] = None,
        valid_file_path: Optional[str] = None,
        output_dir: Optional[str] = None,
        **kwargs,
    ):
        if config_path:
            configs = read_yaml_file(config_path)
            self.model_args = from_dict(
                data_class=ModelArguments, data=configs["ModelArguments"]
            )
            self.data_args = from_dict(
                data_class=DataTrainingArguments, data=configs["DataTrainingArguments"]
            )
            if self.model_args.onnx_mode:
                self.training_args = from_dict(
                    data_class=TRAINING_ARGS_MAP["onnx"],
                    data=configs["TrainingArguments"],
                )
            else:
                self.training_args = from_dict(
                    data_class=TrainingArguments, data=configs["TrainingArguments"]
                )

        else:
            self.model_args = ModelArguments(
                model_name_or_path=model_name_or_path,
                tokenizer_name_or_path=kwargs.get("tokenizer_name_or_path", None),
                cache_dir=kwargs.get("cache_dir", None),
                freeze_embeds=kwargs.get("freeze_embeds", False),
                project_name=kwargs.get("project_name", "question-generation"),
                model_type=kwargs.get("model_type", "t5"),
                model_config=kwargs.get("model_config", None),
                label_smoothing=kwargs.get("label_smoothing", 0),
                onnx_mode=kwargs.get("onnx_mode", False),
            )
            self.data_args = DataTrainingArguments(
                train_file_path=train_file_path,
                valid_file_path=valid_file_path,
                max_source_length=kwargs.get("max_source_length", 512),
                max_target_length=kwargs.get("max_target_length", 128),
            )

            self.training_args = TRAINING_ARGS_MAP[
                "onnx" if kwargs.get("onnx_mode", False) else "default"
            ](
                report_to=kwargs.get("report_to", None),
                per_device_train_batch_size=kwargs.get(
                    "per_device_train_batch_size", 8
                ),
                per_device_eval_batch_size=kwargs.get("per_device_eval_batch_size", 8),
                gradient_accumulation_steps=kwargs.get(
                    "gradient_accumulation_steps", 1
                ),
                learning_rate=kwargs.get("learning_rate", 3.0e-4),
                weight_decay=kwargs.get("weight_decay", 0.0001),
                num_train_epochs=kwargs.get("num_train_epochs", 5),
                lr_scheduler_type=kwargs.get("lr_scheduler_type", "linear"),
                warmup_steps=kwargs.get("warmup_steps", 0),
                warmup_ratio=kwargs.get("warmup_ratio", 0),
                seed=kwargs.get("seed", 42),
                remove_unused_columns=kwargs.get("remove_unused_columns", False),
                evaluation_strategy=kwargs.get("evaluation_strategy", "steps"),
                logging_steps=kwargs.get("logging_steps", 2),
                eval_steps=kwargs.get("eval_steps", 100),
                save_strategy=kwargs.get("save_strategy", "epoch"),
                save_steps=kwargs.get("save_steps", 1000),
                save_total_limit=kwargs.get("save_total_limit", 1),
                load_best_model_at_end=kwargs.get("load_best_model_at_end", True),
                greater_is_better=kwargs.get("greater_is_better", False),
                metric_for_best_model=kwargs.get("metric_for_best_model", "eval_loss"),
                overwrite_output_dir=kwargs.get("overwrite_output_dir", True),
                do_train=kwargs.get("do_train", True),
                do_eval=kwargs.get("do_eval", True),
                auto_find_batch_size=kwargs.get("auto_find_batch_size", False),
                output_dir=output_dir,
            )

    def train(self):
        """
        Training model Q&A Generation model using a configuration file or custom parameters. In the case configuration
        file and custom parameters both passed, this function will prioritize the configuration file.

        Returns:
            Training result will be returned. Training model will be saved at output filepath.

        """
        return start_train(
            model_args=self.model_args,
            data_args=self.data_args,
            training_args=self.training_args,
        )


def start_train(
    model_args: ModelArguments,
    data_args: DataTrainingArguments,
    training_args: Union[TrainingArguments, TRAINING_ARGS_MAP["onnx"]],
):
    assert model_args.model_type in list(
        MODEL_TYPE_TO_TOKENIZER.keys()
    ), "model type should be 't5' or 't5-copy-enhance' or 'bart'"

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. "
            "Use --overwrite_output_dir to overcome."
        )

    if model_args.model_config is not None:
        hyperparameter = load_json_file(model_args.model_config)

    # Setup wandb
    if training_args.report_to == "wandb":
        wandb.login()
        wandb.init(
            project=model_args.project_name,
            name=model_args.model_name_or_path,
            group=model_args.model_type,
            tags=["baseline", "t5"],
            job_type="train",
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters")
    print(training_args)

    # Set seed
    set_seed(training_args.seed)

    # Set project name
    os.environ["WANDB_PROJECT"] = model_args.project_name

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    tokenizer_cls = MODEL_TYPE_TO_TOKENIZER[model_args.model_type]
    tokenizer = tokenizer_cls.from_pretrained(
        model_args.tokenizer_name_or_path
        if model_args.tokenizer_name_or_path
        else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )
    config = AutoConfig.from_pretrained(model_args.model_name_or_path)

    if model_args.model_type == "t5-copy-enhance":
        if model_args.model_config is not None:
            config.update(hyperparameter["t5-enhance"])

        model = T5CopyGenerator.from_pretrained(
            model_args.model_name_or_path, cache_dir=model_args.cache_dir, config=config
        )

    else:
        if model_args.model_config is not None:
            config.update(hyperparameter["t5-base"])

        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_args.model_name_or_path, cache_dir=model_args.cache_dir, config=config
        )

    model.resize_token_embeddings(len(tokenizer))

    if model_args.freeze_embeds:
        logger.info("Freezing embeddings of the model")
        freeze_embeds(model)
        assert_not_all_frozen(model)

    # Get datasets
    logger.info("Loading dataset")

    mix_finetune = False
    if any(
        [
            data_args.train_file_path_new is not None,
            data_args.valid_file_path_new is not None,
        ]
    ):
        mix_finetune = True

        pretrained_train_dataset = (
            torch.load(data_args.train_file_path) if training_args.do_train else None
        )
        new_train_dataset = (
            torch.load(data_args.train_file_path_new)
            if training_args.do_train
            else None
        )
        train_dataset = ConcatDataset([pretrained_train_dataset, new_train_dataset])

        pretrained_valid_dataset = (
            torch.load(data_args.valid_file_path) if training_args.do_eval else None
        )
        new_valid_dataset = (
            torch.load(data_args.valid_file_path) if training_args.do_eval else None
        )
        valid_dataset = ConcatDataset([pretrained_valid_dataset, new_valid_dataset])

        init_mix_finetune = BestRatioMixedFineTune.from_dataset(
            len_pretrained_train_ds=len(pretrained_train_dataset),
            len_new_train_ds=len(new_train_dataset),
            len_total_train_ds=len(train_dataset),
        )
        ratio = init_mix_finetune.get_best_ratio()
        weights = torch.tensor(
            [
                ratio / len(pretrained_train_dataset)
                if i < len(new_train_dataset)
                else (1 - ratio) / len(new_train_dataset)
                for i in range(len(train_dataset))
            ]
        )

        print("*** best ratio for mixed finetune ***", ratio.item())
        print(
            "*** rate b/t pretrained_dataset and new_dataset ***",
            len(pretrained_train_dataset) / len(new_train_dataset),
        )
        print("*** estimated rate ***", (weights[-1] / weights[0]).item())
        torch.save(weights, "data/weights.pt")

    else:
        train_dataset = (
            torch.load(data_args.train_file_path) if training_args.do_train else None
        )
        valid_dataset = (
            torch.load(data_args.valid_file_path) if training_args.do_eval else None
        )

    logger.info("Finished loading dataset")

    # Initialize data_collator
    data_collator = Text2TextDataCollator(
        tokenizer=tokenizer,
        model_type=model_args.model_type,
        mode="training",
        using_tpu=training_args.tpu_num_cores is not None,
    )

    # Initialize our Trainer
    trainer = TRAINER_ARGS_MAP["onnx" if model_args.onnx_mode else "default"](
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=data_collator,
        mix_finetune=mix_finetune,
    )

    # Disable wandb console logs
    logging.getLogger("wandb.run_manager").setLevel(logging.WARNING)

    # Training
    if training_args.do_train:
        trainer.train(
            model_path=model_args.model_name_or_path
            if os.path.isdir(model_args.model_name_or_path)
            else None
        )
        trainer.save_model()
        # For convenience, we also re-save the tokenizer to the same directory,
        # so that you can share your model easily on huggingface.co/models =)

        # if trainer.is_world_master():
        tokenizer.save_pretrained(training_args.output_dir)

    # Evaluation
    results = {}
    if training_args.do_eval and training_args.local_rank in [-1, 0]:
        logger.info("*** Evaluate ***")

        eval_output = trainer.evaluate()

        output_eval_file = os.path.join(training_args.output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results *****")
            for key in sorted(eval_output.keys()):
                logger.info("  %s = %s", key, str(eval_output[key]))
                writer.write("%s = %s\n" % (key, str(eval_output[key])))

        results.update(eval_output)

    return results


def main(args_file=None):
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments)
    )

    if (len(sys.argv) == 2 and sys.argv[1].endswith(".json")) or args_file is not None:
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        args_file_path = (
            os.path.abspath(sys.argv[1]) if args_file is None else args_file
        )
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=args_file_path
        )
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    return start_train(
        model_args=model_args, data_args=data_args, training_args=training_args
    )


if __name__ == "__main__":
    set_gpu(2)

    main()
