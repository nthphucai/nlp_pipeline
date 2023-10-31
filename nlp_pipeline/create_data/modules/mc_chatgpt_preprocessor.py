import string
from typing import List, Tuple

from tqdm import tqdm

from mongodb.mongo_client import connect_mongodb
from questgen.utils.file_utils import get_time, logger, read_yaml_file


class ChatGPTPreprocessor:
    def __init__(self, task: str, domain: str, database_config_path: str):
        """
        ChatGPTPreprocessor class.
        Args:
            task (str): Task of preprocessing data.
            domain (str): Domain of preprocessing data.
            database_config_path (str): Path of database config.
        """
        self.task = task
        self.domain = domain
        self.database_config_path = database_config_path
        self.crawl_database = None
        self.preprocess_database = None
        self.answers = "ABCDEFGH"
        self.__initialize()

    def __initialize(self):
        self.crawl_database = connect_mongodb(
            self.database_config_path, "crawl_collection_name"
        )
        self.preprocess_database = connect_mongodb(
            self.database_config_path, "preprocess_collection_name"
        )
        config = read_yaml_file(self.database_config_path)
        self.answer_template = config["chatgpt_preprocessor"]["domain"][self.domain]
        self.correct_templates = config["chatgpt_preprocessor"]["correct_templates"]

    def __get_data_from_database(self) -> List[dict]:
        """
        Get data from database.
        Returns:
            List[str]: List of data.
        """
        data = list(
            self.crawl_database.find(
                {"domain": self.domain, "task": self.task, "source": "chatgpt"}
            )
        )
        return [item["data"] for item in data]

    def __extract_qa_item(self, data: List[dict]) -> Tuple[int, int]:
        """
        Extract question-answer information from each item of data.
        Args:
            data (List[dict]): List of crawled data.

        Returns:
            Tuple[int, int]: Number of extracted qa item, Number of completed qa item.
        """
        count = 0
        success = 0

        for item in tqdm(data, desc="Extracting qa item: "):
            qa_item = []
            for line in item["crawl"].split("\n"):
                line = line.strip()
                if len(line) > 0:
                    if line[-1] in ":?":
                        if len(qa_item) > 1:
                            if self.__format_and_log_qa_item(
                                {"context": item["context"], "content": qa_item}
                            ):
                                success += 1
                            count += 1
                        qa_item = [line]
                    else:
                        qa_item.append(line)
            if len(qa_item) > 1:
                if self.__format_and_log_qa_item(
                    {"context": item["context"], "content": qa_item}
                ):
                    success += 1
                count += 1
        return count, success

    def __format_and_log_qa_item(self, item: dict) -> bool:
        """
        Format question-answer data and log it into database.
        Args:
            item (dict): A dictionary of crawled qa item.

        Returns:
            bool: QA item can be formatted or not.
        """
        output = {
            "domain": self.domain,
            "task": self.task,
            "context": item["context"],
            "question": item["content"][0],
        }
        options = []
        answer = None

        for line in item["content"][1:]:
            if self.answer_template.lower() not in line[:6].lower():
                while line[-1] == "." and line[-3:] != "...":
                    line = line[:-1]
                if len(line) > 2:
                    if (
                        line[0].lower() in self.answers.lower()
                        and line[1] in string.punctuation
                    ):
                        line = line[2:].strip()
                if len(line) > 0:
                    options.append(line)
            else:
                if ":" in line:
                    answer = line.split(":")[1].strip()
                    if len(answer) <= 2:
                        answer = answer[0].upper()
                    else:
                        if (
                            answer[0].lower() in self.answers.lower()
                            and answer[1] in string.punctuation
                        ):
                            answer = answer[0].upper()
                        else:
                            while answer[-1] == "." and answer[-3:] != "...":
                                answer = answer[:-1]
                            if answer in options and len(options) > 0:
                                answer = self.answers[options.index(answer)]
                            else:
                                answer = None

        if answer is None and len(options) > 0:
            for option in options:
                for template in self.correct_templates:
                    if template.lower() in option.lower():
                        answer = self.answers[options.index(option)]
                        options[options.index(option)] = option.replace(
                            template, ""
                        ).strip()
                        break

        if answer is not None and len(options) > 0:
            output["options"] = options
            output["answer"] = answer
            output["time"] = get_time()
            self.preprocess_database.insert_one(output)
            return True
        return False

    def preprocessing_data(self):
        """
        Preprocessing crawled data.
        """
        data = self.__get_data_from_database()
        logger.info(f"# {len(data)} crawled item")

        count, success = self.__extract_qa_item(data)
        logger.info(f"# {count} extracted item; # {success} formated item;")
