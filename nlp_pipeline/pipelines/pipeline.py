# coding=utf-8
import sys
import warnings
from typing import Optional, Union

from transformers import (
    AutoModelForSeq2SeqLM,
    PreTrainedModel,
    PreTrainedTokenizer,
    T5Tokenizer,
    set_seed,
)


sys.path.append("utils")
warnings.filterwarnings("ignore")

from questgen.pipelines.mc_pipeline import MCPipeline
from questgen.pipelines.multitask_pipeline import MultiTaskPipeline
from questgen.utils.file_utils import read_yaml_file


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


def pipeline(
    task: str,
    model: Optional[Union[str, PreTrainedModel]] = None,
    tokenizer: Optional[Union[str, PreTrainedTokenizer]] = None,
    config_path: str = None,
    **kwargs,
):
    # Read the configuration for question generation pipeline
    config = read_yaml_file(config_path)
    print(f"Generation Pipeline `{config['name']}` config:\n{config}")

    set_seed(21)

    # Retrieve the task
    if task not in SUPPORTED_TASKS:
        raise KeyError(
            f"Unknown task {task}, available tasks are {list(SUPPORTED_TASKS.keys())}"
        )

    targeted_task = SUPPORTED_TASKS[task]
    task_class = targeted_task["impl"]

    # Use default model/config/tokenizer for the task if no model is provided
    if model is None:
        model = targeted_task["default"]["model"]

    # Try to infer tokenizer from model or config name (if provided as str)
    if tokenizer is None:
        if isinstance(model, str):
            tokenizer = model
        else:
            # Impossible to guest what is the right tokenizer here
            raise Exception(
                "Impossible to guess which tokenizer to use. "
                "Please provided a PretrainedTokenizer class or a path/identifier to a pretrained tokenizer."
            )

    # Instantiate tokenizer if needed
    if isinstance(tokenizer, (str, tuple)):
        if isinstance(tokenizer, tuple):
            # For tuple we have (tokenizer name, {kwargs})
            tokenizer = T5Tokenizer.from_pretrained(tokenizer[0], **tokenizer[1])
        else:
            tokenizer = T5Tokenizer.from_pretrained(tokenizer)

    # Instantiate model if needed
    if isinstance(model, str):
        model = AutoModelForSeq2SeqLM.from_pretrained(model)

    if task == "question-generation":
        return task_class(model=model, tokenizer=tokenizer, **config["generate_qa"])

    elif task == "multitask":
        return task_class(model=model, tokenizer=tokenizer, **config["generate_qa"])

    elif task == "multiplechoice":
        return task_class(
            num_options=4,
            model=model,
            tokenizer=tokenizer,
            **config["generate_distractors"],
        )
