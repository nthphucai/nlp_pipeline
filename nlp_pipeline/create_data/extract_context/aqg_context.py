import gc
from types import ModuleType
from typing import Dict, List, Union

from overrides import override
from termcolor import colored

from questgen.create_data.extract_context.context_base import ContextExtracterBase
from questgen.utils.file_utils import logger


class CreateData(ContextExtracterBase):
    def __init__(
        self, context: List[str], preprocessor: ModuleType, save_path: str, config: dict
    ):
        super().__init__(context, preprocessor, **config)
        self.context = context
        self.preprocessor = preprocessor
        self.save_path = save_path

    def __call__(self, qa_data):
        result = self.extract_context(qa_data)
        return result

    def extract_context_one_item(self, data: Dict) -> Dict:
        """
        Extract context by item.
        Args:
            data (Dict): An data item.

        Returns:
            Dict: An context added data item.
        """
        right_option = data["options"][self.list_answers.index(data["answer"])]
        preprocessed_option = self.preprocessor.get_all_lowered_words(right_option)
        best_relevant = self._find_relevant_content(
            data["question"], preprocessed_option
        )

        if best_relevant is not None:
            best_relevant_context = best_relevant["best_relevant_context"]
            match_answer = self._extract_match_answer(
                right_option, best_relevant_context
            )[1]
            answer_start = self.find_answer_start(match_answer, best_relevant_context)
            data["context"] = best_relevant_context
            data["match_answer"] = match_answer
            data["answer_start"] = answer_start

        else:
            data["context"] = None
            data["match_answer"] = None
            data["answer_start"] = -1

        if self.verbose:
            print(colored("question:", "blue"), colored(data["question"], "white"))
            print(colored("answer:", "blue"), colored(right_option, "white"))
            if best_relevant is not None:
                print(
                    colored("match_question_rate:", "blue"),
                    best_relevant["match_question_rate"],
                )
                print(
                    colored("match_answer:", "green"),
                    colored(data["match_answer"], "green"),
                )

        gc.collect()
        del best_relevant
        return data


class CreateDataBM25Search(ContextExtracterBase):
    def __init__(
        self, context: List[str], preprocessor: ModuleType, save_path: str, config: dict
    ):
        super().__init__(context, preprocessor, **config)
        self.preprocessor = preprocessor
        self.save_path = save_path
        self.context = context

    def __call__(self, qa_data):
        result = self.extract_context(qa_data)
        return result

    def extract_context_one_item(self, data: Dict) -> Dict:
        """
        Extract context by item.
        Args:
            data (Dict): An data item.

        Returns:
            Dict: An context added data item.
        """
        try:
            right_option = data["options"][self.list_answers.index(data["answer"])]
            preprocessed_option = self.preprocessor.get_all_lowered_words(right_option)

            relevant_context = self._find_relevant_content_with_bm25search(
                data["question"], preprocessed_option
            )
            if len(relevant_context) > 0:
                relevant_context = relevant_context[0]
                data["context"] = relevant_context
                data["match_answer"] = right_option
                data["answer_start"] = self.find_answer_start(
                    right_option, relevant_context
                )

            else:
                data["context"] = None
                data["match_answer"] = right_option
                data["answer_start"] = -1

            gc.collect()
            del relevant_context

        except Exception as e:
            logger.error(e)
            data = None

        return data

    @override
    def _expand_context(
        self, question: str, retrieval_context: str, threshold=0.85
    ) -> Union[str, None]:
        """
        Expand the context by question information.
        Args:
            question (str): Content of question.
            retrieval_context (str): Content of retrieval context.
            threshold (float): Min target matching rate from question to context.

        Returns:
            Union[str, None]: The expanded context or None.
        """
        wsegmented_question = self.preprocessor.word_segment(question)
        wsegmented_context = self.preprocessor.word_segment(retrieval_context)
        begin = self.context.index(retrieval_context) - 1
        end = begin + 2
        flag = True
        while (len(retrieval_context.split(" ")) < 1000) and (
            self._calculate_matching_rate(wsegmented_question, wsegmented_context)
            < threshold
        ):
            if (flag) and (end < len(self.context)):
                retrieval_context += " " + self.context[end]
                end += 1
            elif (not flag) and (begin >= 0):
                retrieval_context = self.context[begin] + " " + retrieval_context
                begin -= 1
            flag = not flag
            wsegmented_context = self.preprocessor.word_segment(retrieval_context)
        if (
            self._calculate_matching_rate(wsegmented_question, wsegmented_context)
            >= threshold
        ):
            return retrieval_context
        return None

    def _find_relevant_content_with_bm25search(
        self, question: str, answer: str
    ) -> List[str]:
        """
        Find relevant content for question-answer pair.
        Args:
            question (str): Content of question.
            answer (str): Content of answer.

        Returns:
            List[str]: List of relevant context.
        """
        finded_item = []

        for document in self.preprocessed_context:
            if answer in document:
                document = self.context[self.preprocessed_context.index(document)]
                document = self._expand_context(question, document)
                if document is not None:
                    finded_item.append(document.strip())
        if len(finded_item) > 1:
            relevant_context = self.bm25_search(question, finded_item)
        elif len(finded_item) == 1:
            relevant_context = finded_item
        else:
            relevant_context = []
        return [item.strip() for item in relevant_context]
