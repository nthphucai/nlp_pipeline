import re
import warnings
from random import shuffle
from typing import Generator, List

import numpy as np
import spacy
import yake

from pyvi import ViTokenizer


class DistractorBase:
    def __init__(self, kw_extract_strategy, n_grams):
        self.strategy = kw_extract_strategy
        self.n_grams = n_grams

        self.extract_kw = KeywordExtractor(self.strategy, self.n_grams)

    def masking(self, sentence: str, num_masks: int = 2) -> str:
        """
        Replace keywords by mask_token.

        Args:
            sentence: (`str`) - Text need to be masked
            num_masks: (`int`) - Number of mask tokens

        Returns: masked sentences
        """
        keywords = self.extract_kw(sentence, num_masks)
        for kw in keywords:
            if "_" in kw:
                kw = " ".join(kw.split("_"))
            sentence = sentence.replace(kw, self.mask_token, 1)

        sentence = ViTokenizer.tokenize(sentence)
        sentence = sentence.replace("< ", "<").replace(" >", ">")
        return sentence

    def unmasking(
        self,
        org_sent: str,
        masked_sent: str,
        pred: str,
        index: int = 0,
        stop: bool = False,
    ):
        max_value = max([len(pred[num_mask]) for num_mask in range(len(pred))])
        if index >= max_value:
            return self.unmasking(
                org_sent, masked_sent, pred, index=index - 1, stop=True
            )

        word_lst = masked_sent.split()
        num_mask = 0
        for idc, word in enumerate(word_lst):
            if word == "<mask>":
                if index < len(pred[num_mask]):
                    word_lst[idc] = self._normalize_predict_word(
                        pred[num_mask][index]["token_str"]
                    )
                    num_mask += 1
                else:
                    word_lst[idc] = self._normalize_predict_word(
                        pred[num_mask][0]["token_str"]
                    )

        result = " ".join(word_lst)
        predicted_sent = self._normalize_predict_sentence(result)
        if stop:
            return predicted_sent
        matching_score = self.calculate_matching_rate(org_sent, predicted_sent)
        return (
            self.unmasking(org_sent, masked_sent, pred, index=index + 1)
            if matching_score > 0.95
            else predicted_sent
        )

    def unmasking_multiple_results(
        self, masked_sent: str, pred: List[dict], mask_idx: int = 0, score: float = 1
    ):
        """
        Unmasking multiple distractors and get the full list of result
        by combination of probability score.

        Args:
            masked_sent (str), pred (List[dict])
            mask_idx (int, optional): The corresponding index. Defaults to 0.
            score (float, optional): The corresponding probability. Defaults to 1.

        Returns:
            List[dict]: A list of distractors.
        """

        results = []
        for predict_word in pred[mask_idx]:
            predicted_sentence = self._preplace_first_mask(
                masked_sent, predict_word["token_str"]
            )
            if mask_idx == len(pred) - 1:
                results.extend(
                    [
                        {
                            "predict_sentence": self._normalize_predict_sentence(
                                predicted_sentence
                            ),
                            "score": score * predict_word["score"],
                        }
                    ]
                )
            else:
                results.extend(
                    self.unmasking_multiple_results(
                        masked_sent=predicted_sentence,
                        pred=pred,
                        mask_idx=mask_idx + 1,
                        score=score * predict_word["score"],
                    )
                )
        return results

    def _preplace_first_mask(self, masked_sent: str, predict_word: str):
        return masked_sent.replace("<mask>", predict_word, 1)

    def _normalize_predict_word(self, word):
        removed_space_word = word.replace(" ", "")
        return " ".join(removed_space_word.split("_"))

    def _normalize_predict_sentence(self, sentence):
        removed_special_token = (
            sentence.replace("<s>", "").replace("</s>", "").replace("@@", "")
        )
        unsegment_text = " ".join(removed_special_token.split("_"))
        return re.sub(r"\s+", " ", unsegment_text).strip()

    def calculate_matching_rate(
        self, question_seg: str, context_seg: str, stopwords_path=None
    ) -> float:
        question_seg = question_seg.split(" ")
        context_seg = context_seg.split(" ")

        if stopwords_path is not None:
            stopwords = list(np.load(stopwords_path))
            question_seg = [word for word in question_seg if word not in stopwords]

        count = 0
        for item in question_seg:
            if item in context_seg:
                count += 1
        return count / len(question_seg)


class KeywordExtractor:
    """
    Description:
        KeywordExtractor class extract keywords from question
        with two types of strategies "pos_tagging" and "Yake".
    Return:
        keywords extracted from question.
    """

    def __init__(self, strategy: str = "pos_tagging", n_grams=2):
        self.model = None
        self._init_extract_model(strategy, n_grams=n_grams)

    def _init_extract_model(self, strategy: str, n_grams: int):
        if strategy == "pos_tagging":
            self.model = spacy.load("vi_core_news_lg")
        elif strategy == "yake":
            self.model = yake.KeywordExtractor(n=n_grams)
        else:
            self.model = spacy.load("vi_core_news_lg")
            warnings.warn(
                "strategy not in [`pos_tagging`, `yake`] - Used `pos_tagging` as default"
            )

    def __call__(self, sentence, num_keywords_return=2) -> List[str]:
        keywords = list(self._extract_keywords(sentence))
        shuffle(keywords)
        return keywords[:num_keywords_return]

    def _extract_keywords(self, sentence: str) -> Generator:
        doc = self.model(sentence)
        for token in doc:
            if (token.tag_ == "N" or token.tag_ == "V") and token.text.count("_") == 1:
                yield token.text
