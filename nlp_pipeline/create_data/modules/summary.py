import copy
from types import ModuleType
from typing import List

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from termcolor import colored

import regex
from gensim.models import KeyedVectors
from pyvi import ViTokenizer
from questgen.create_data.extract_context.context_base import ContextExtracterBase
from questgen.utils.constants import STOPWORD_PATH, VI_VECTOR_PATH
from questgen.utils.file_utils import logger


class SummaryContext(ContextExtracterBase):
    """
    Description:
        Create ...
    Return:
        Dataset ...
    """

    def __init__(
        self,
        preprocessor: ModuleType,
        type_task: str = "mc",
        max_context_length: int = 1500,
        rate_cluster: float = 0.6,
        threshold: float = 0.6,
        verbose: bool = True,
    ):
        self.preprocessor = preprocessor
        self.type_task = type_task
        self.max_context_length = max_context_length
        self.rate_cluster = rate_cluster
        self.threshold = threshold
        self.verbose = verbose

        self.w2v = KeyedVectors.load_word2vec_format(VI_VECTOR_PATH)
        self.vocab = self.w2v.wv.vocab

        self.mapping_answer = {"A": 0, "B": 1, "C": 2, "D": 3}
        self.stopwords_path = STOPWORD_PATH

    def __call__(self, examples: dict):
        example = copy.deepcopy(examples)

        context = example["context"]
        question = example["question"]
        match_answer = example["answers"]["text"][0]
        correct_answer_index = self.mapping_answer[example["answers"]["correct_option"]]
        correct_answer = example["options"][correct_answer_index]

        question, correct_answer, match_answer = [
            item.lower() for item in [question, correct_answer, match_answer]
        ]
        summary_context, best_summary_context = [context] * 2
        condition = lambda context: match_answer.lower() in context.lower()

        cnt = 0
        while len(summary_context) >= self.max_context_length:
            summary_context = self._summary_context(summary_context, self.rate_cluster)
            result = self._find_relevant_one_summary_context(
                correct_answer, question, summary_context
            )
            max_rate = result["max_rate"]
            cnt += 1
            if cnt == 20:
                break
            elif condition(summary_context):
                if (match_answer == correct_answer) and (max_rate >= 0.6):
                    self.rate_cluster -= 0.1
                    best_summary_context = summary_context
                elif max_rate >= self.threshold:
                    self.rate_cluster -= 0.1
                    best_summary_context = summary_context
                else:
                    break
            else:
                break

        if self.verbose:
            print("question:", question)
            print("answer:", correct_answer)
            print("match_answer:", match_answer)
            print(colored(len(context), "blue"))
            print(colored(len(best_summary_context), "green"))
            print("\n")

        assert condition(best_summary_context), "answer must be in summary context"
        example["context"] = best_summary_context
        example["answer_start"] = self._find_answer_start(
            match_answer, best_summary_context
        )
        return example

    def _find_relevant_one_summary_context(self, answer, question, summary_context):
        question, answer, summary_context = [
            item.lower() for item in [question, answer, summary_context]
        ]
        max_rate = 0
        answer = self.preprocessor.word_segment(answer)
        question = self.preprocessor.word_segment(question)

        assert [
            item.islower() for item in [question, answer, summary_context]
        ], "input must be lowered"

        max_rate = self._calculate_matching_rate(
            question, summary_context, stopwords=True
        )
        return {
            "max_rate": max_rate,
            "question": question,
            "answer": answer,
            "best_relevant_context": summary_context,
        }

    def _word2vec(self) -> list:
        temp_lst = []
        for sentence in self.sentences:
            sentence = ViTokenizer.tokenize(sentence)
            words = sentence.split(" ")
            sentence_vec = np.zeros((100))
            for word in words:
                if word in self.vocab:
                    sentence_vec += self.w2v.wv[word]
            temp_lst.append(sentence_vec)
        return temp_lst

    def _prepare_inputs_for_summary(self, sentences: str) -> List[str]:
        sentences = self.sentence_segment(sentences)
        result = [sent for sent in sentences if sent is not None]
        return result

    def _summary_context(self, context, rate_cluster) -> str:
        self.sentences = self._prepare_inputs_for_summary(context)
        sentence_length = len(self.sentences)

        n_clusters = int(sentence_length * rate_cluster)
        if n_clusters < 3:
            n_clusters = int(sentence_length)
        self.kmeans = KMeans(n_clusters)
        temp_lst = self._word2vec()
        kmeans = self.kmeans.fit(temp_lst)
        avg = []
        try:
            for j in range(n_clusters):
                idx = np.where(kmeans.labels_ == j)[0]
                avg.append(np.mean(idx))

            closest, _ = pairwise_distances_argmin_min(
                kmeans.cluster_centers_, temp_lst
            )
            ordering = sorted(range(n_clusters), key=lambda k: avg[k])
            summary = ". ".join([self.sentences[closest[idx]] for idx in ordering])
            return summary

        except Exception as e:
            logger.info(e)

    @staticmethod
    def sentence_segment(text):
        sents = regex.split("([.?!])?[\n]+|[.?!] ", text)
        return sents
