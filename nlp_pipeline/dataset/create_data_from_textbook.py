import argparse
import pathlib
from itertools import chain

from questgen.create_data.extract_context.aqg_context import (
    CreateData,
    CreateDataBM25Search,
)
from questgen.create_data.modules.preprocessor import Preprocessor
from questgen.create_data.modules.summary import SummaryContext
from questgen.utils import READ_FILE_FN
from questgen.utils.file_utils import (
    load_json_file,
    logger,
    read_text_file,
    read_yaml_file,
    write_json_file,
)
from questgen.utils.utils import multiprocess


SUPPORTED_TASKS = {
    "summary": SummaryContext,
    "preprocess": Preprocessor,
    "extract-context": CreateData,
    "extract-context-bm25": CreateDataBM25Search,
}


def prepare_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        type=str,
        default="summary",
        help="only support 'summary', 'preprocess', 'extract-context', 'extract-context-bm25'",
    )
    parser.add_argument(
        "--qa_pair_data_path", type=str, help="Path for question - answer pairs data."
    )
    parser.add_argument("--context_data_path", type=str, help="Path for context data.")
    parser.add_argument("--AQG_PATH", type=str, help="Path to save SAQG result.")
    parser.add_argument("--AQGSUM_PATH", type=str, help="Path to save SAQG result.")

    parser.add_argument(
        "--output_dir", type=str, default="data", help="Path for output data directory."
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default="configs/create_qa_data_config.yml",
        help="Path for creating question answering data config.",
    )
    parser.add_argument(
        "--use_multiprocessing",
        type=bool,
        default=False,
        help="Use multi processing or not.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=16,
        help="Number of workers when using multiprocessing.",
    )
    return parser.parse_args()


def create_context(
    task: str,
    qa_pair_data_path: str,
    context_data_path: str,
    num_workers: int,
    config_path: str,
    AQG_PATH: str,
    AQGSUM_PATH: str,
):
    logger.info("Loading data")
    file_extension = pathlib.Path(qa_pair_data_path).suffix
    qa_data = READ_FILE_FN[file_extension](qa_pair_data_path, convert_to_json=False)[
        :100
    ]

    data_path = {"SAQG_path": AQG_PATH}
    data_sum_path = {"SAQG_path": AQGSUM_PATH}

    # Read the configuration for question generation pipeline
    config = read_yaml_file(config_path)
    preprocessor = SUPPORTED_TASKS["preprocess"](config)

    if "extract-context" in task:
        print(
            f"Create Dataset Pipeline {config['name']} config:\n{config['dataset_task']}"
        )
        context = read_text_file(context_data_path)

        targeted_task = SUPPORTED_TASKS[task]
        task_class = targeted_task(
            context, preprocessor, data_path, config["dataset_task"]
        )
        result = task_class(qa_data)
        logger.info("total sa_samples: %s", len(list(chain(*result))))
        logger.info("AQG_test_module saved at %s ", data_path["SAQG_path"])

    if task == "summary":
        print(
            f"Create Summary Pipeline {config['name']} config:\n{config['summary_task']}"
        )

        targeted_task = SUPPORTED_TASKS[task]
        summary_inputs = targeted_task(
            preprocessor, "SAQG_path", **config["summary_task"]
        )
        data = load_json_file(data_path["SAQG_path"])["data"]
        results = multiprocess(
            iter_func=summary_inputs, iter_args=data, workers=num_workers, disable=False
        )

        logger.info("total samples %s ", len(results))
        logger.info("Module saved at %s ", data_sum_path["SAQG_path"])
        write_json_file({"data": results}, data_sum_path["SAQG_path"])


def main():
    args = prepare_parser()
    create_context(
        task=args.task,
        qa_pair_data_path=args.qa_pair_data_path,
        context_data_path=args.context_data_path,
        AQG_PATH=args.AQG_PATH,
        AQGSUM_PATH=args.AQGSUM_PATH,
        num_workers=args.num_workers,
        config_path=args.config_path,
    )


if __name__ == "__main__":
    main()
