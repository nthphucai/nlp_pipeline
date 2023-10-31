import copy
import re
from abc import ABC, abstractmethod
from itertools import chain
from typing import Dict, Iterable, List, Optional

import numpy as np
import yake
from rank_bm25 import BM25Okapi
from termcolor import colored
from torch.utils.data import DataLoader as load_batch
from tqdm import tqdm, trange

from questgen.create_data.modules.preprocessor import Preprocessor
from questgen.utils.constants import STOPWORD_PATH
from questgen.utils.file_utils import logger, write_json_file
from questgen.utils.utils import multiprocess


class ContextExtracterBase(ABC):
    def __init__(
        self,
        context: List,
        preprocessor: Preprocessor = None,
        data_columns_name: dict = None,
        num_workers: int = 20,
        batch_size: int = 200,
        question_upper_bound: float = 0.8,
        prefix_upper_bound: float = 0.85,
        answer_lower_bound: float = 0.55,
        answer_upper_bound: float = 0.95,
        max_context_length: int = 1500,
        max_num_context: int = 1,
        verbose: bool = False,
    ) -> None:
        self.preprocessor = preprocessor
        self.workers = num_workers
        self.batch_size = batch_size
        self.max_num_context = max_num_context
        self.list_answers = list("ABCD")
        self.verbose = verbose

        self.question_upper_bound = question_upper_bound
        self.prefix_upper_bound = prefix_upper_bound
        self.answer_lower_bound = answer_lower_bound
        self.answer_upper_bound = answer_upper_bound
        self.max_context_length = max_context_length

        self.stopwords_path = STOPWORD_PATH

        self.context = context
        self.preprocessed_context = [
            self.preprocessor.get_all_lowered_words(item)
            for item in tqdm(context, desc="Preprocessing context: ")
        ]
        self.data_columns_name = data_columns_name

        self._initialize()

    def _initialize(self):
        self.map_right_answer = {}
        for item in self.list_answers:
            self.map_right_answer[item] = (
                self.data_columns_name["options_prefix"] + item
            )

    def extract_context(self, qa_data):
        """
        Description: Create Dataset from History Text Book
        """
        logger.info("Preparing data")
        self.context = self._prepare_context(self.context)
        qa_data = self._prepare_qa_data(qa_data)

        logger.info("Extracting data context")

        result = []
        for id in trange(len(qa_data), desc="Preprocessing options: "):
            qa_data[id]["options"] = [
                self.preprocessor.preprocess_answer(option)
                for option in qa_data[id]["options"]
            ]

        logger.info("Extracting context")
        data_batch = load_batch(
            qa_data,
            batch_size=self.batch_size,
            num_workers=self.workers,
            collate_fn=self._generate_batch,
        )
        for _, dloader in tqdm(enumerate(data_batch), total=len(data_batch)):
            try:
                output_data = multiprocess(
                    iter_func=self.extract_context_one_item,
                    iter_args=dloader,
                    workers=self.workers,
                    disable=True,
                )
            except Exception as e:
                logger.error(e)
                continue

            output_data = [item for item in output_data if item["answer_start"] >= 0]
            saqg_data = self.format_aqg_data(output_data)
            result.append(saqg_data)

            result = [item for item in result if item != []]
            flat_answers = lambda answer_lst: list(chain(*answer_lst))
            qg_out = flat_answers(result)
            write_json_file({"data": qg_out}, self.save_path["SAQG_path"])

        return result

    @staticmethod
    def _generate_batch(batch):
        batch_dict = [
            {
                "question": example["question"],
                "answer": example["answer"],
                "options": example["options"],
            }
            for example in batch
        ]
        return batch_dict

    def _prepare_context(self, context: List[str]) -> List[str]:
        """
        Normalize information has number in context
        Args:
            context (List[str]): List of context.

        Returns:
            List[str]: List of normalized context.
        """
        documents = []
        for document in context:
            words = []
            for word in document.split(" "):
                if len(words) > 0 and words[-1].isdigit() and word.isdigit():
                    words[-1] += word
                else:
                    words.append(word)
            documents.append(
                self.preprocessor.process_accent_by_document(" ".join(words))
            )
        return documents

    def _prepare_qa_data(self, df) -> List[dict]:
        """
        Convert dataframe has QA data to list of dictionary.
        Args:
            df (pd.DataFrame): Dataframe of QA data.

        Returns:
            List[dict]: List of QA data dictionary.
        """

        def prepare_qa_item(id: int) -> Dict:
            df_item = df.iloc[id]
            qa_item = {}

            if (
                str(df_item[self.data_columns_name["right_answer_name"]])
                in self.list_answers
            ) and (str(df_item[self.data_columns_name["right_answer_name"]]) != "nan"):
                qa_item["question"] = self.preprocessor.process_accent_by_document(
                    df_item[self.data_columns_name["question_name"]]
                )
                qa_item["answer"] = df_item[self.data_columns_name["right_answer_name"]]
                qa_item["options"] = []

                for answer in self.list_answers:
                    if str(df_item[self.map_right_answer[answer]]) != "nan":
                        qa_item["options"].append(
                            self.preprocessor.process_accent_by_document(
                                df_item[self.map_right_answer[answer]]
                            )
                        )
            return qa_item

        qa_data = list(map(prepare_qa_item, trange(len(df), desc="Preparing data: ")))
        qa_data = [item for item in qa_data if item is not None]
        qa_data = [
            item
            for item in qa_data
            if ("options" in item and len(item["options"]) == 4)
        ]
        qa_data = [item for item in qa_data if len(item.keys()) > 0]
        return qa_data

    def bm25_search(self, question: str, list_context: List[str]) -> str:
        context_seg = [
            self.preprocessor.word_segment(context).split(" ")
            for context in list_context
        ]
        question_seg = self.preprocessor.word_segment(question).split(" ")

        bm25 = BM25Okapi(context_seg)
        top_context = bm25.get_top_n(question_seg, context_seg, n=self.max_num_context)
        del bm25
        return [list_context[context_seg.index(context)] for context in top_context]

    def _calculate_matching_rate(
        self, question_seg: str, context_seg: str, stopwords: bool = False
    ) -> float:
        """
        Calculate the matching rate from word-segmented question to word-segmented context.
        Args:
            wsegmented_question (str): Content of word-segmented question.
            wsegmented_context (str): Content of word-segmented context.

        Returns:
            float: The matching rate from question to context.
        """
        question_seg = question_seg.split(" ")
        context_seg = context_seg.split(" ")

        if stopwords:
            stopwords = list(np.load(self.stopwords_path))
            question_seg = [word for word in question_seg if word not in stopwords]

        count = 0
        for item in question_seg:
            if item in context_seg:
                count += 1
        return count / len(question_seg)

    def _expand_context(
        self, question: str, retrieval_context: str, threshold=0.85
    ) -> Optional[str]:
        question_seg = self.preprocessor.word_segment(question)
        context_seg = self.preprocessor.word_segment(retrieval_context)

        begin = self.context.index(retrieval_context) - 1
        end = begin + 2
        flag = True
        while (len(retrieval_context.split(" ")) < 1000) and (
            self._calculate_matching_rate(question_seg, context_seg, stopwords=True)
            < threshold
        ):
            if (flag) and (end < len(self.context)):
                retrieval_context += " " + self.context[end]
                end += 1
            elif (not flag) and (begin >= 0):
                retrieval_context = self.context[begin] + " " + retrieval_context
                begin -= 1
            flag = not flag
            context_seg = self.preprocessor.word_segment(retrieval_context)

        if (
            self._calculate_matching_rate(question_seg, context_seg, stopwords=False)
            >= threshold
        ):
            return retrieval_context
        return None

    def _find_relevant_content(self, question: str, answer: str, t=0.85):
        """
        Description:
            Find relevant context w.r.t question
        Return:
            Origin Relevant Content
        """
        step = 0
        match_question_rate = 0
        longest_answer = "a"
        relevant_answer_dict = None
        best_answer_relevant = None

        relevant_answer_dict = self._search_relevant_prefix_answer(question, answer)
        if relevant_answer_dict is None:
            return None

        relevant_preprocessed_context = relevant_answer_dict[
            "relevant_preprocessed_context"
        ]
        for document in relevant_preprocessed_context:
            sub_answer = self._extract_match_answer(answer, document)[1]

            if sub_answer is not None:
                if len(sub_answer) > len(longest_answer):
                    longest_answer = sub_answer
                    best_answer_relevant = document

        if best_answer_relevant is None:
            return None

        start_index = self.preprocessed_context.index(best_answer_relevant)
        end_index = start_index

        question_seg = self.preprocessor.word_segment(question)
        stopwords = list(np.load(self.stopwords_path))
        question_seg = [
            word for word in question_seg.split(" ") if word not in stopwords
        ]
        question_seg = " ".join(question_seg)

        while match_question_rate < t:
            start_index = start_index - 1
            end_index = end_index + 1

            best_relevant_context = self.preprocessed_context[
                start_index : (end_index + 1)
            ]
            best_relevant_context = ".".join(best_relevant_context)
            context_seg = self.preprocessor.word_segment(best_relevant_context)
            match_question_rate = self._calculate_matching_rate(
                question_seg, context_seg, stopwords=True
            )
            match_question_rate = np.round(match_question_rate, 1)
            step += 1
            if len(best_relevant_context) <= self.max_context_length:
                if (longest_answer.lower() == answer.lower()) and (
                    match_question_rate > 0.6
                ):
                    match_question_rate = self.question_upper_bound
                    break
                elif step > 100:
                    t -= 0.05
                    continue
                elif (step > 200) and (t < self.question_upper_bound):
                    break
            else:
                break

        if match_question_rate < self.question_upper_bound:
            return None

        # get best_relevant_context w.r.t index
        relevant_context = self.context[start_index : (end_index + 1)]
        relevant_context = [item.strip() for item in relevant_context]
        best_relevant_context = " ".join(relevant_context)

        return {
            "best_relevant_context": best_relevant_context,
            "match_question_rate": match_question_rate,
        }

    def _get_next_word(self, query_sent) -> Iterable:
        sent = "_"
        for step in range(len(query_sent)):
            sent += "_" + query_sent[step]
            sent = sent.replace("_", " ")
            yield sent

    def _get_best_sentences(self, key: str) -> Iterable:
        kwords_seg = self.preprocessor.word_segment(key).split(" ")
        query_sent = kwords_seg

        for idc in range(len(kwords_seg)):
            concat_word = self._get_next_word(query_sent)
            query_sent = kwords_seg[idc:]
            for best_sentence in concat_word:
                yield best_sentence

    def _extract_match_answer(self, key_lower: str, context_lower: str) -> str:
        max_len = 0
        best_sentence = None

        key_lower, context_lower = [item.lower() for item in [key_lower, context_lower]]

        if key_lower in context_lower:
            best_sentence = key_lower
            return 1, best_sentence, None

        sentences = self._get_best_sentences(key_lower)
        for _, sent in enumerate(sentences):
            sent_cp = copy.deepcopy(sent)
            sent_cp = sent_cp.strip()
            if sent_cp in context_lower:
                if len(sent_cp) > max_len:
                    max_len = len(sent_cp)
                    best_sentence = sent_cp

        if best_sentence is not None:
            remain_key = re.sub("\s+", " ", key_lower.replace(best_sentence, ""))
            rate = len(best_sentence) / len(key_lower)
            return rate, best_sentence, remain_key
        else:
            return 0, None, key_lower

    def _extract_kw_question(self, question, ratio, stop_path=None):
        stopwords = list(np.load(stop_path)) if stop_path is not None else None
        n_gram = int(len(re.findall("\s", question)) / ratio)
        kw_extractor = yake.KeywordExtractor(lan="vi", n=n_gram, stopwords=stopwords)
        try:
            keywords = kw_extractor.extract_keywords(question)[0][0]
            return keywords

        except Exception:
            return question

    def _get_best_yake_ratio(self, keyword, ratio) -> List[str]:
        """
        Description: Fine tune to get best ratio
                    for yake w.r.t keyword (question)
        Args:
            Inputs: keyword, initial ratio
            Return: best ratio w.r.t question
        """
        assert isinstance(keyword, str), "the question must be string"

        prefix_rate = 0
        start_index = 0

        for idc, doc in enumerate(self.preprocessed_context):
            document = self.preprocessed_context[start_index : (idc + 1)]
            document = ".".join(document)

            if prefix_rate < self.prefix_upper_bound:
                start_index = self.preprocessed_context.index(doc) - 1
                prefix = self._extract_kw_question(keyword, ratio, self.stopwords_path)
                prefix_rate, _, _ = self._extract_match_answer(prefix, document)

            if prefix_rate != 0:
                best_ratio = ratio
                return best_ratio

    def _search_relevant_prefix_answer(
        self, question, answer, ratio=3, cnt=0, prefix: bool = False
    ) -> List[str]:
        """
        Description: Find relevant context w.r.t qa(answer, question)
        Args:
            Inputs: answers need to query in context
            Return: List of relevant context w.r.t answer based on threshold condition
        """
        assert isinstance(answer, str), "the answer must be string"
        assert isinstance(question, str), "the question must be string"

        rate, cnt, prefix_rate, start_index = [0] * 4
        flag = False
        result = None
        relevant_answer_context = []
        self.best_answer_set = set()

        if ratio == 10:
            return None

        if not prefix:
            ratio = self._get_best_yake_ratio(question, ratio)
            if ratio is None:
                return None
            prefix = self._extract_kw_question(question, ratio).lower()

        self.prefix_answer = prefix + " " + answer

        for idc, doc in tqdm(
            enumerate(self.preprocessed_context),
            total=len(self.preprocessed_context),
            disable=True,
        ):
            document = self.preprocessed_context[start_index : (idc + 1)]
            document = ".".join(document)

            if (flag is False) and (prefix_rate < self.prefix_upper_bound):
                start_index = self.preprocessed_context.index(doc) - 1
                prefix_rate, best_prefix, _ = self._extract_match_answer(
                    prefix, document
                )
                if prefix_rate > self.prefix_upper_bound:
                    flag = True

            elif flag is True:
                cnt += 1
                if cnt > 1000:
                    return None

                elif rate < self.answer_lower_bound:
                    self.best_answer_set.add(best_prefix)
                    rate, best_answer, remain_kw = self._extract_match_answer(
                        answer, document
                    )

                elif self.answer_lower_bound < rate < 1:
                    self.best_answer_set.add(best_answer)
                    accum_rate, best_answer, remain_kw = self._extract_match_answer(
                        remain_kw, document
                    )
                    if best_answer is None:
                        continue

                    rate += accum_rate
                    self.best_answer_set.add(best_answer)
                    self.best_answer_set = set(
                        [
                            answer
                            for answer in self.best_answer_set
                            if answer is not None
                        ]
                    )
                    full_answer = " ".join(self.best_answer_set)
                    matching_rate = len(full_answer) / len(self.prefix_answer)
                    if matching_rate < self.answer_upper_bound:
                        continue
                    else:
                        end_index = self.preprocessed_context.index(doc) + 1
                        relevant_answer_context = self.preprocessed_context[
                            start_index : (end_index + 1)
                        ]
                        relevant_context = [
                            context.strip() for context in relevant_answer_context
                        ]
                        result = {"relevant_preprocessed_context": relevant_context}
                        break
                else:
                    self.best_answer_set.add(best_answer)
                    end_index = self.preprocessed_context.index(doc) + 1
                    relevant_answer_context = self.preprocessed_context[
                        start_index : (end_index + 1)
                    ]
                    relevant_context = [
                        context.strip() for context in relevant_answer_context
                    ]
                    result = {"relevant_preprocessed_context": relevant_context}
                    break
            else:
                return None

        if (self.verbose) and (result is not None):
            print(
                colored("\nbest_answer_set:", "green"),
                colored(self.best_answer_set, "green"),
            )

        if result is None:
            ratio += 1
            result = self._search_relevant_prefix_answer(question, answer, ratio, cnt)
            return result

        return result

    @staticmethod
    def find_answer_start(
        answer: str, best_relevant_context: str, match_all=True
    ) -> int:
        """
        Find the index of answer's appearance in context.
        Args:
            answer (str): Content of answer.
            context (str): Content of context.
            match_all (bool): Word matching from answer to context is absolutely of not.

        Returns:
            int: The index of answer's appearance in context.
        """
        lowered_answer = answer.lower()
        lowered_context = best_relevant_context.lower()
        answer_start = []
        if not match_all:
            while answer_start == -1 and len(lowered_answer) > 0:
                if lowered_answer in lowered_context:
                    return lowered_context.index(lowered_answer)
                lowered_answer = " ".join(lowered_answer.split(" ")[:-1])
        else:
            if lowered_answer in lowered_context:
                return lowered_context.index(lowered_answer)
        return answer_start

    @staticmethod
    def highlight_words(sentence: str, words: str):
        for w in words:
            pos = sentence.lower().find(w.lower())
            sentence = (
                sentence
                if (
                    (pos < 0) or (sentence[pos - 1] != " " and sentence[pos + 1] != " ")
                )
                else sentence[0:pos]
                + "[*"
                + w.upper()
                + "*]"
                + sentence[pos + len(w) :]
            )
        return sentence

    def format_aqg_data(self, data: List[dict] = None) -> List[dict]:
        """
        Format for the simple question generation data output.
        Args:
            data (List[dict]): List of question-answer information dictionary.
            id_format (str): Format for ID parameter data output.
            title (str): Title of data.

        Returns:
            List[dict]: List of formatted question-answer information dictionary for simple question generation data.
        """
        output_data = []
        mapping_answer = {"A": 0, "B": 1, "C": 2, "D": 3}

        for id in trange(len(data), desc="Formatting SAQG Data: "):
            str_id = "0" * 6 + str(id + 1)
            answer_index = mapping_answer[data[id]["answer"]]
            temp_dict = {
                "context": data[id]["context"],
                "id": "fschool_lichsu_saqg" + str_id[-6:],
                "question": data[id]["question"],
                "title": "Lịch Sử",
                "answers": {
                    "answer_start": [data[id]["answer_start"]],
                    "text": [data[id]["match_answer"]],
                    "correct_answer": [data[id]["options"][answer_index]],
                    "correct_option": data[id]["answer"],
                },
                "options": data[id]["options"],
            }
            output_data.append(temp_dict)
        return output_data

    @abstractmethod
    def extract_context_one_item(self, data):
        pass
