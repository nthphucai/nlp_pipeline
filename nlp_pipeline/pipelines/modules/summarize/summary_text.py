import os
import warnings
from pathlib import Path
from typing import List

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min

import questgen
from gensim.models import KeyedVectors
from pyvi import ViTokenizer
from questgen.pipelines.modules.preprocess import __mapping__ as preprocess


warnings.filterwarnings("ignore")

PIPELINE_PATH = os.path.join(Path(questgen.__file__).parent, "pipelines")
VI_VECTOR_PATH = os.path.join(PIPELINE_PATH, "modules/summarize/vi.vec")


class TextSummarize:
    def __init__(self, sentences: List[str], rate_cluster=0.6):
        self.w2v = KeyedVectors.load_word2vec_format(VI_VECTOR_PATH)
        self.vocab = self.w2v.wv.vocab

        self.sentences = self._prepare_inputs_for_summary(sentences)
        sentence_length = len(self.sentences)

        self.n_clusters = int(sentence_length * rate_cluster)
        if self.n_clusters < 3:
            self.n_clusters = int(sentence_length)

        self.kmeans = KMeans(n_clusters=self.n_clusters)

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
        sentences = preprocess["viquad_noise"](sentences)
        sentences = preprocess["hle"](sentences)
        sentences = preprocess["sent_segment"](sentences)
        result = [sent for sent in sentences if sent is not None]
        return result

    def __call__(self) -> str:
        try:
            temp_lst = self._word2vec()
            kmeans = self.kmeans.fit(temp_lst)
            avg = []
            for j in range(self.n_clusters):
                idx = np.where(kmeans.labels_ == j)[0]
                avg.append(np.mean(idx))

            closest, _ = pairwise_distances_argmin_min(
                kmeans.cluster_centers_, temp_lst
            )
            ordering = sorted(range(self.n_clusters), key=lambda k: avg[k])
            summary = ". ".join([self.sentences[closest[idx]] for idx in ordering])
            return summary

        except AssertionError:
            pass
