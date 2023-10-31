import re
from typing import List

from tqdm import tqdm

from pyvi.ViTokenizer import tokenize
from questgen.create_data.modules.accent_processor import AccentProcessor
from questgen.utils.file_utils import logger, read_yaml_file


class Preprocessor(AccentProcessor):
    def __init__(self, config) -> None:
        super().__init__()

        self.config = read_yaml_file(config)
        self._initialize()

    def _initialize(self):
        self.convert_mapping = self.config["convert_mapping"]
        if "bm25_top_k" in self.config:
            self.top_k = self.config["bm25_top_k"]
        else:
            self.top_k = 1

    def preprocess_pdf_context(self, context: List[str]) -> List[str]:
        """
        Preprocess pdf extracted context.
        Args:
            context (List[str]): List of pdf extracted context.

        Returns:
            List[str]: List of preprocessed pdf extracted context.
        """

        def remove_noise(text: str) -> str:
            """
            Remove noise in text.
            Args:
                text (str): Input text.

            Returns:
                str: Removed noise text.
            """
            for key, value in self.convert_mapping.items():
                text = re.sub(key, value, text)

            while " ," in text:
                text = text.replace(" ,", ",")
            while " )" in text:
                text = text.replace(" )", ") ")
            while "( " in text:
                text = text.replace("( ", " (")

            return text.strip()

        def format_paragraph(context: List[str]) -> List[str]:
            """
            Concatenate context to paragraph.
            Args:
                context (List[str]): List of context.

            Returns:
                List[str]: List of paragraph.
            """
            formatted_context = []
            flag = True
            for item in tqdm(context):
                if flag:
                    formatted_context.append(item[0].upper() + item[1:])
                    if item[-1] != "." or item[-1] == "!":
                        flag = False
                else:
                    if item[0] == "-":
                        formatted_context.append(item)
                    else:
                        formatted_context[-1] += " " + item
                    if item[-1] == ".":
                        flag = True
            return formatted_context

        logger.info("Removing noise")
        context = [
            remove_noise(item)
            for item in tqdm(context, desc="Removing noise context: ")
        ]
        context = [item for item in context if item != ""]
        context = [self.process_accent_by_document(item) for item in context]
        logger.info("Formating context paragraph")
        context = format_paragraph(context)
        return context

    def get_all_lowered_words(self, document: str) -> str:
        """
        Get all lowercased word and number in document.
        Args:
            document (str): Input document.

        Returns:
            str: All lowercased word and number in document (remove punct, special character...)
        """
        words = re.findall("\w+", document)
        return " ".join(words).strip().lower()

    def word_segment(self, text: str) -> str:
        """
        Segment text and remove all punctuations, special tokens...
        Args:
            text (str): Input text

        Returns:
            str: Word segmented text.
        """
        wsegmented_text = tokenize(text)
        wsegmented_text = self.get_all_lowered_words(wsegmented_text)
        return wsegmented_text

    def preprocess_answer(self, answer: str) -> str:
        """
        Preprocess answer content.
        Args:
            answer (str): Input answer content.

        Returns:
            str: Preprocessed answer content.
        """
        answer = answer.replace("“", "").replace("”", "")
        while answer[-1] in ".":
            answer = answer[:-1]
        answer = answer.replace("/", "-")
        answer = answer.replace("-", " - ")
        answer = answer.replace("…", "")
        for key, value in self.convert_mapping.items():
            answer = answer.replace(key, value)
        answer = re.sub(r"\s+", " ", answer)
        return answer.strip()
