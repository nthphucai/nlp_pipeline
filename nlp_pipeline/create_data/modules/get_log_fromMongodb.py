import argparse
from typing import List

from pymongo import MongoClient
from tqdm import tqdm

from questgen.utils.file_utils import logger, write_json_file
from questgen.utils.format_utils import format_saqg_data


POST_PROCESS_WORD = ["Tại vì", "Vì", "Tại", "Để", "Do"]


def prepare_args():
    """
    Reading argument from input.
    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", type=str, help="Path for output json")
    parser.add_argument(
        "--connection_string",
        type=str,
        help="Connection string to connect to database.",
    )
    parser.add_argument("--database_name", type=str, help="Name of database.")
    parser.add_argument(
        "--collection_name", type=str, help="Name of database's collection."
    )
    return parser.parse_args()


def get_log_data(collection) -> List[dict]:
    """
    Get web-app demo logged data.
    Args:
        collection: Collection mongodb.

    Returns:
        List[dict]: Logged data.
    """
    log_data = []
    for item in tqdm(list(collection.find()), desc="Getting log data: "):
        if len(item["results"]) > 0:
            temp_dict = {"context": item["context"]}
            for qa_info in item["results"]:
                temp_dict["question"] = qa_info["question"]
                temp_dict["answer"] = qa_info["answer"]
                if qa_info["comment"] != "":
                    comment = qa_info["comment"]
                    if "[SEP]" in comment:
                        question, answer = comment.split("[SEP]")
                        temp_dict["question"] = question.split("Question:")[1].strip()
                        temp_dict["answer"] = question.split("Answer:")[1].strip()
                    elif "Question:" in comment:
                        temp_dict["question"] = comment.split("Question:")[1].strip()
                    elif "Answer:" in comment:
                        temp_dict["answer"] = comment.split("Answer:")[1].strip()
                elif qa_info["check"] == 0:
                    continue
                log_data.append(temp_dict)
    return log_data


def preprocess_answer(answer: str) -> str:
    """
    Preprocess from answer after post-processing in web-app.
    Args:
        answer (str): Post-processed answer content.

    Returns:
        str: Preprocessed post-processed answer content.
    """
    for prefix in POST_PROCESS_WORD:
        if prefix in answer:
            if answer.index(prefix) == 0:
                answer = answer.replace(prefix, "")
                return answer
    return answer.strip()


def convert_to_saqg_data(log_data: List[dict]) -> List[dict]:
    """
    Convert log data to saqg data.
    Args:
        log_data (List[dict]): Log data.
        extracter (ContextExtracter): ContextExtracter module.

    Returns:
        List[dict]: SAQG data.
    """
    saqg_data = []

    find_answer_start = lambda context, answer: context.lower().index(answer.lower())

    for item in tqdm(log_data, desc="Finding answer start: "):
        preprocessed_answer = preprocess_answer(item["answer"]).strip()
        temp = {
            "context": item["context"],
            "question": item["question"],
            "answer": list([preprocessed_answer]),
            "answer_start": find_answer_start(preprocessed_answer, item["context"]),
        }
        if temp not in saqg_data:
            saqg_data.append(temp)

    saqg_data = format_saqg_data(
        data=saqg_data, id_format="log_lichsu_saqg_", title="Lịch sử"
    )

    return saqg_data


def main():
    args = prepare_args()
    output_path = args.output_path
    connection_string = args.connection_string
    database_name = args.database_name
    collection_name = args.collection_name

    logger.info("Preparing database")
    client = MongoClient(connection_string)
    collection = client[database_name][collection_name]
    log_data = get_log_data(collection)
    client.close()

    logger.info("Converting to SAQG data format")
    saqg_data = convert_to_saqg_data(log_data)
    logger.info("Length of log_data: ", len(saqg_data))

    write_json_file({"data": saqg_data}, output_path)
    logger.info(f"Output data at: {output_path}")


if __name__ == "__main__":
    main()
