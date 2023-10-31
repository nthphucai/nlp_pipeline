# coding=utf-8
import copy
import warnings
from dataclasses import dataclass, field
from typing import Optional, Union

import pandas as pd
from pandas import DataFrame as df
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    HfArgumentParser,
    PreTrainedModel,
    PreTrainedTokenizer,
)


warnings.filterwarnings("ignore")

from questgen.pipelines.multitask_pipeline import MultiTaskPipeline
from questgen.pipelines.sum_pipeline import SummaryPipeline
from questgen.utils.file_utils import (
    load_json_file,
    logger,
    read_yaml_file,
    write_json_file,
)


SUPPORTED_TASKS = {
    "summary": {"impl": SummaryPipeline, "default": {"model": "output/t5-base-qg-hl"}},
    "multitask": {
        "impl": MultiTaskPipeline,
        "default": {"model": "output/t5-base-multitask-hl"},
    },
}


@dataclass
class InferArguments:
    data_path: str = field(
        metadata={"help": "Path to data to create more training_data"}
    )
    model_name_or_path: str = field(
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        }
    )

    config_path: str = field(metadata={"help": "Path to config yaml file"})

    tokenizer_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained tokenizer name or path if not the same as model_name"
        },
    )

    save_path: Optional[str] = field(
        default=None, metadata={"help": "Whether to save sum_data"}
    )

    update_data: bool = field(
        default=False,
        metadata={
            "help": "Whether to combine train_viquad and sum_data into new_train_viquad"
        },
    )

    use_summary: bool = field(
        default=False, metadata={"help": "whether to load data from save_path"}
    )

    sort_data: bool = field(
        default=False, metadata={"help": "whether to load data from save_path"}
    )

    format_for_enhance: bool = field(
        default=False, metadata={"help": "whether to format for enhance data "}
    )


def inferpipeline(
    task: str,
    model: Optional[PreTrainedModel] = None,
    tokenizer: Optional[Union[str, PreTrainedTokenizer]] = None,
    config_path: str = None,
    **kwargs,
):
    targeted_task = SUPPORTED_TASKS[task]
    task_class = targeted_task["impl"]

    if task == "summary":
        task = task_class()
        return task

    elif task == "multitask":
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

        task = task_class(model=model, tokenizer=tokenizer, **config["generation_task"])
        return task


"""
Prepare data for training
"""


def convert_to_dict(idc, result) -> dict:
    row = result.iloc[idc]

    columns = ["context", "summary", "id", "title", "question", "answer"]
    context, summary, id_, title, question, answer = row[[*columns]]

    start_c = summary.lower().find(answer.lower())
    end_c = start_c + len(answer)
    text = summary[start_c:end_c]

    new_dict = dict()
    new_dict["org_context"] = context
    new_dict["context"] = summary
    new_dict["id"] = id_
    new_dict["question"] = question
    new_dict["title"] = title
    new_dict["answers"] = {"answer_start": [start_c], "text": [text]}
    return new_dict


def main():
    parser = HfArgumentParser((InferArguments,))
    args = parser.parse_args_into_dataclasses()[0]

    context = load_json_file(path=args.data_path)["data"]
    logger.info("The numbers of new data is %s", len(context))

    if args.use_summary:
        summary = inferpipeline(task="summary")
        context = summary(context)
        logger.info("Finished summarizing data")

        if args.save_path is not None:
            write_json_file(data=context, path="data/json/sum_data_test.json")

    else:
        assert isinstance(context, list)
        assert isinstance(context[0], dict)

        multitask = inferpipeline(
            task="multitask",
            model=args.model_name_or_path,
            config_path=args.config_path,
        )
        qa, inputs = multitask(context)

        assert len(qa) == len(
            inputs
        ), "the length of qa_pairs must be equal to the length of inputs"

        if args.format_for_enhance:
            data = pd.concat([df.from_dict(inputs), df({"qa": qa})], axis=1)

            if "summary" not in data.columns:
                data["summary"] = data["context"]

            data = data.rename(
                columns={"question": "question_phase1", "answers": "answers_phase1"}
            )

            columns = [
                "context",
                "id",
                "question_phase1",
                "title",
                "answers_phase1",
                "summary",
            ]

            temp_lst = []
            for idc in range(len(data)):
                row = data.loc[idc, "qa"]
                col = data.iloc[[idc]][[*columns]]
                df_expand = col.loc[col.index.repeat(len(row))].reset_index(drop=True)
                new_data = pd.concat([df_expand, df([*row])], axis=1)
                temp_lst.append(new_data)

            result = pd.concat(temp_lst).reset_index(drop=True)
            data_lst = [
                convert_to_dict(idc, result=result) for idc in range(result.shape[0])
            ]

            print("...exclude non-answer extraction task...")
            update_data = []
            for idc, d in enumerate(data_lst):
                if d["answers"]["answer_start"][0] != -1:
                    update_data.append(d)

            if args.update_data:
                viquad_data = load_json_file(path=args.data_path)["data"]
                logger.info("The numbers of train_viquad %s", len(viquad_data))
                update_data.extend(viquad_data)

            if args.sort_data:
                logger.info("...sorting data before saving...")
                cp_ = copy.deepcopy(update_data)
                context = [cp_[idc]["context"] for idc in range(len(cp_))]
                indexes_sorted = [
                    value[0]
                    for value in sorted(
                        enumerate(context), key=lambda item: len(item[1])
                    )
                ]
                update_data = [cp_[idc] for idc in indexes_sorted]

            if args.save_path is not None:
                new_dict = dict()
                new_dict["data"] = update_data

                write_json_file(data=new_dict, path=args.save_path)
                logger.info("Saved sum_train_viquad at %s", args.save_path)

            logger.info("The numbers of sum_train_viquad is %s", len(update_data))


if __name__ == "__main__":
    main()
