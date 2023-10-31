# coding=utf-8
import json
import os
import pathlib
from dataclasses import dataclass, field
from typing import List, Optional, Union

import torch
from torch.multiprocessing import set_start_method
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    HfArgumentParser,
    PreTrainedModel,
    PreTrainedTokenizer,
    T5Tokenizer,
)

from questgen.pipelines.mc_pipeline import MCPipeline
from questgen.pipelines.multitask_pipeline import MultiTaskPipeline
from questgen.utils import READ_FILE_FN
from questgen.utils.constants import BOOK_PATH
from questgen.utils.file_utils import (
    download_trained_model,
    load_json_file,
    logger,
    read_yaml_file,
    write_json_file,
)
from questgen.utils.utils import set_gpu


SUPPORTED_TASKS = {
    "multitask": {
        "impl": MultiTaskPipeline,
        "default": {"model": "output/t5-base-multitask-hl"},
    },
    "multiplechoice": {
        "impl": MCPipeline,
        "default": {"model": "output/t5-base-multitask-hl"},
    },
}


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[
        0
    ]  # First element of model_output contains all token embeddings
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )


@dataclass
class InferArguments:
    data_path: str = field(
        metadata={"help": "Path to data to create more training_data"}
    )

    task: str = field(metadata={"help": "'multitask', 'multiplechoice'"})

    multitask_model_name_or_path: str = field(
        metadata={
            "help": "Path to pretrained model or model from huggingface.co/models"
        }
    )

    mc_model_name_or_path: str = field(
        default=None,
        metadata={
            "help": "Path to pretrained model or model from huggingface.co/models"
        },
    )

    config_aqg_path: str = field(
        default=None, metadata={"help": "Path to config multitask-mc yaml file"}
    )

    tokenizer_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained tokenizer name or path if not the same as model_name"
        },
    )

    save_path_multitask: Optional[str] = field(
        default=None, metadata={"help": "Path to save multitask_data"}
    )

    save_path_mc: Optional[str] = field(
        default=None, metadata={"help": "Path to save mc_data"}
    )

    update_data: bool = field(
        default=False,
        metadata={
            "help": "Whether to combine train_viquad and sum_data into new_train_viquad"
        },
    )

    use_summary: bool = field(
        default=False, metadata={"help": "whether to use summary"}
    )

    use_multiprocess: bool = field(
        default=False, metadata={"help": "whether to use use_multiprocess"}
    )

    only_distractors: bool = field(
        default=False, metadata={"help": "whether to generate only distrators"}
    )

    download_model: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where do you want to store the pretrained models downloaded"
        },
    )


def inferpipeline(
    use_multiprocess: bool = False,
    task: str = "multitask",
    model: Optional[PreTrainedModel] = None,
    tokenizer: Optional[Union[str, PreTrainedTokenizer]] = None,
    config_path: str = None,
    **kwargs,
):
    targeted_task = SUPPORTED_TASKS[task]
    task_class = targeted_task["impl"]
    if task == "multitask":
        config = read_yaml_file(config_path)
        print(f"Generation Pipeline `{config['name']}` config:\n{config}")

        if tokenizer is None:
            if isinstance(model, str):
                tokenizer = model
            else:
                # Impossible to guest what is the right tokenizer here
                raise Exception(
                    "Impossible to guess which tokenizer to use. "
                    "Please provided a PretrainedTokenizer class or a path/identifier to a pretrained tokenizer."
                )

        if isinstance(tokenizer, (str, tuple)):
            if isinstance(tokenizer, tuple):
                # For tuple we have (tokenizer name, {kwargs})
                tokenizer = AutoTokenizer.from_pretrained(tokenizer[0], **tokenizer[1])
            else:
                tokenizer = AutoTokenizer.from_pretrained(tokenizer)

        if isinstance(model, str):
            model = AutoModelForSeq2SeqLM.from_pretrained(model)

        task = task_class(
            use_multiprocess=use_multiprocess,
            model=model,
            tokenizer=tokenizer,
            **config["generate_qa"],
        )

    elif task == "multiplechoice":
        config = read_yaml_file(config_path)
        print(f"Generation Pipeline `{config['name']}` config:\n{config}")

        if tokenizer is None:
            if isinstance(model, str):
                tokenizer = model
            else:
                # Impossible to guest what is the right tokenizer here
                raise Exception(
                    "Impossible to guess which tokenizer to use. "
                    "Please provided a PretrainedTokenizer class or a path/identifier to a pretrained tokenizer."
                )

        if isinstance(tokenizer, (str, tuple)):
            if isinstance(tokenizer, tuple):
                # For tuple we have (tokenizer name, {kwargs})
                tokenizer = AutoTokenizer.from_pretrained(tokenizer[0], **tokenizer[1])
            else:
                tokenizer = T5Tokenizer.from_pretrained(tokenizer)

        if isinstance(model, str):
            model = AutoModelForSeq2SeqLM.from_pretrained(model)

            task = task_class(
                use_multiprocess=use_multiprocess,
                num_options=4,
                model=model,
                tokenizer=tokenizer,
                **config["generate_distractors"],
            )

    return task


class Inference:
    def __init__(
        self,
        multitask_model_name_or_path: str,
        config_aqg_path: str,
        mc_model_name_or_path: Optional[str] = None,
        tokenizer_name_or_path: Optional[str] = None,
        only_distractors: bool = False,
        download_model: str = None,
        use_summary: bool = False,
        use_multiprocess: bool = False,
    ):
        self.config_aqg_path = config_aqg_path
        self.tokenizer_name_or_path = tokenizer_name_or_path
        self.only_distractors = only_distractors
        self.download_model = download_model
        self.use_summary = use_summary
        self.use_multiprocess = use_multiprocess

        if download_model:
            if multitask_model_name_or_path:
                download_trained_model(
                    domain=download_model,
                    save_path=multitask_model_name_or_path,
                    task="multitask",
                )
            if mc_model_name_or_path:
                download_trained_model(
                    domain=download_model, save_path=mc_model_name_or_path, task="mc"
                )

        self.multitask_model_name_or_path = multitask_model_name_or_path
        self.mc_model_name_or_path = mc_model_name_or_path
        if multitask_model_name_or_path:
            self.multitask_pipeline = inferpipeline(
                use_multiprocess=self.use_multiprocess,
                task="multitask",
                model=self.multitask_model_name_or_path,
                tokenizer=self.tokenizer_name_or_path,
                config_path=self.config_aqg_path,
            )
        if mc_model_name_or_path:
            self.mc_pipeline = inferpipeline(
                use_multiprocess=self.use_multiprocess,
                task="multiplechoice",
                model=self.mc_model_name_or_path,
                tokenizer=self.tokenizer_name_or_path,
                config_path=self.config_aqg_path,
            )
        with open(BOOK_PATH) as f:
            data = json.load(f)
            self.book = data

    @staticmethod
    def _prepare_data(context: Union[str, List[str]] = None, data_path: str = None):
        example_lst = []
        if context:
            if isinstance(context, str):
                context = [context]
            for ctx in context:
                example_lst.append({"context": ctx, "org_context": ctx})
        elif data_path:
            file_extension = pathlib.Path(data_path).suffix
            examples = READ_FILE_FN[file_extension](data_path)
            example_lst = examples["data"] if ".json" in file_extension else examples
        else:
            raise Exception("Both context and data_path were not specified")
        logger.info("The numbers of data is %s", len(example_lst))
        return example_lst

    @staticmethod
    def flatten(list_of_lists):
        if len(list_of_lists) == 0:
            return list_of_lists
        if isinstance(list_of_lists[0], list):
            return Inference.flatten(list_of_lists[0]) + Inference.flatten(
                list_of_lists[1:]
            )
        return list_of_lists[:1] + Inference.flatten(list_of_lists[1:])

    @staticmethod
    def concat_context(data):
        if isinstance(data, dict):
            context = [Inference.concat_context(value) for value in data.values()]
            return context
        return [data]

    def parse_data_from_book(self, path):
        path = path.split(">")
        data = self.book
        for key in path:
            data = data[key]
        data = Inference.concat_context(data)
        return self.flatten(data)

    def generate_qa_pair_from_book(self, task, path="Lop>Chuong>Bai>Muc"):
        context = self.parse_data_from_book(path)
        qa_pair = self.create(task=task, context=context)
        return qa_pair

    def create(
        self,
        task: str = "multitask",
        context: Union[str, List[str]] = None,
        save_path_multitask: str = None,
        save_path_mc: str = None,
        data_path: str = None,
        lang: str = "vi",
    ):
        if not os.path.exists(os.path.dirname(save_path_multitask)):
            os.makedirs(os.path.dirname(save_path_multitask))

        examples = self._prepare_data(context=context, data_path=data_path)
        if (task == "multitask") or (task == "mc" and not self.only_distractors):
            qa_pair = self.multitask_pipeline(
                examples=examples, use_summary=self.use_summary
            )
            if save_path_multitask is not None:
                write_json_file({"data": qa_pair}, save_path_multitask)
                logger.info(f"Infer_multitask saved at {save_path_multitask}")
                logger.info("The numbers of multitask data is %s", len(qa_pair))

        if "mc" in task:
            if self.only_distractors:
                print(
                    f"... Loading existing multitask data at {save_path_multitask}..."
                )
                qa_pair = load_json_file(save_path_multitask)["data"]
            qa_pair = self.mc_pipeline(examples=qa_pair, lang=lang)
            qa_pair = [item for item in qa_pair if len(item["answers"]) >= 4]

            if save_path_mc is not None:
                if not os.path.exists(os.path.dirname(save_path_mc)):
                    os.makedirs(os.path.dirname(save_path_mc))

                write_json_file({"data": qa_pair}, save_path_mc)
                logger.info(f"Infer_mc saved at {save_path_mc}")
                logger.info("The numbers of multiplechoice data is %s", len(qa_pair))

        return qa_pair


def main():
    parser = HfArgumentParser((InferArguments,))
    args = parser.parse_args_into_dataclasses()[0]

    # setting gpu
    set_gpu(0, 1)

    # allow re-initialize cuda in forked subprocess
    set_start_method("spawn")

    inference = Inference(
        multitask_model_name_or_path=args.multitask_model_name_or_path,
        mc_model_name_or_path=args.mc_model_name_or_path,
        config_aqg_path=args.config_aqg_path,
        only_distractors=args.only_distractors,
        download_model=args.download_model,
        use_summary=args.use_summary,
        use_multiprocess=args.use_multiprocess,
    )

    inference.create(
        task=args.task,
        context=None,
        data_path=args.data_path,
        save_path_multitask=args.save_path_multitask,
        save_path_mc=args.save_path_mc,
    )


if __name__ == "__main__":
    main()
