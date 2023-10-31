import copy
import json
import logging
from typing import Iterable

import datasets
import nltk

nltk.download("punkt")

_DESCRIPTION = """\Introduction about dataset.
"""

_CITATION = """\
"""

QG_FORMATS = [
    "prepend",
    "highlight",
    "prepend_highlight",
]
class ViquadQGConfig(datasets.BuilderConfig):
    """BuilderConfig for ViQuAD-QG Datasets."""

    def __init__(self, qg_format="highlight", sub_task="multitask", **kwargs):
        """BuilderConfig for ViQuAD-QG.
        Args:
            **kwargs: keyword arguments forwarded to super.
        """
        super(ViquadQGConfig, self).__init__(**kwargs)
        self.qg_format = qg_format
        self.sub_task = sub_task


class ViquadQG(datasets.GeneratorBasedBuilder):
    """ViQuAD-QG: A Vietnamese Dataset for Question Generation. Version 1.1."""

    BUILDER_CONFIGS = [
        ViquadQGConfig(
            name=f"{format_}_qg_format",
            version=datasets.Version(
                "1.1.0", "New split API (https://tensorflow.org/datasets/splits)"
            ),
            description="Plain text",
            qg_format=format_,
            sub_task=ViquadQGConfig().sub_task,
        )
        for format_ in QG_FORMATS
    ]


    def _info(self):
        features = datasets.Features(
            {
                "source_text": datasets.Value("string"),
                "target_text": datasets.Value("string"),
                "task": datasets.Value("string"),
            }
        )

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            supervised_keys=None,
            homepage="https://huggingface.co/datasets/thanhns/viquad_qg/tree/main/",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        urls_to_download = {
            "train": self.config.data_files["train"][0],
            "validation": self.config.data_files["validation"][0],
            "test": self.config.data_files["test"][0],
        }

        downloaded_files = dl_manager.download_and_extract(urls_to_download)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"filepath": downloaded_files["train"]},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={"filepath": downloaded_files["validation"]},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"filepath": downloaded_files["test"]},
            ),
        ]

    def _get_correct_alignment(self, context, gold_text, start_idx):
        """
        Some original examples in ViQuADv1.1 have indices wrong by 1 or 2 character. We test and fix this here.
        """
        end_idx = start_idx + len(gold_text)
        context, gold_text = context.lower(), gold_text.lower()

        # When the gold label position is good
        if context[start_idx:end_idx] == gold_text:
            return start_idx, end_idx

        # When the gold label is off by one character
        elif context[start_idx - 1 : end_idx - 1] == gold_text:
            return start_idx - 1, end_idx - 1

        # When the gold label is off by two character
        elif context[start_idx - 2 : end_idx - 2] == gold_text:
            return start_idx - 2, end_idx - 2

        # When the gold label is off by one character
        elif context[start_idx + 1 : end_idx + 1] == gold_text:
            return start_idx + 1, end_idx + 1

    def process_qg_text(
        self,
        context: str,
        question: str,
        answer: dict,
    ):
        answer_text = answer["text"][0]
        start_idx = answer["answer_start"][0]

        if self.config.qg_format == "prepend":
            question_gen_input = f"answer: {answer_text} context: {context}"
        elif self.config.qg_format == "highlight":
            start_pos, end_pos = self._get_correct_alignment(
                context, answer_text, start_idx
            )
            question_gen_input = f"generate question: {context[:start_pos]} {{hl_token}} {context[start_pos: end_pos]} {{hl_token}} {context[end_pos:]}"
        else:
            start_pos, end_pos = self._get_correct_alignment(
                context, answer_text, start_idx
            )
            question_gen_input = f"answer: {context[start_pos: end_pos]} context: {context[:start_pos]} {{hl_token}} {context[start_pos: end_pos]} {{hl_token}} {context[end_pos:]}"

        question_gen_target = f"{question}"

        examples = {
            "source_text": question_gen_input,
            "target_text": question_gen_target,
            "task": "qg",
        }

        return examples

    def process_qa_text(self, context, question, answer):
        answer_text = answer["text"][0].strip()
        answer_gen_input = f"question: {question} context: {context}"
        answer_gen_target = f"{answer_text}"

        examples = {
            "source_text": answer_gen_input,
            "target_text": answer_gen_target,
            "task": "qa",
        }

        return examples

    def process_answer_extraction(self, article):
        context = article["context"].strip()
        sentences = nltk.sent_tokenize(context)

        examples = []
        source_text = "extract_answer: "
        for ans in article["answers"]["text"]:
            ans_lower = ans.lower()
            sents = copy.deepcopy(sentences)
            for idc, sent in enumerate(sents):
                sent_lower = sent.lower()
                if ans_lower in sent_lower:
                    sents[idc] = f"{{hl_token}} {sent} {{hl_token}}"

            input_text = f"{source_text}" + " ".join(sents)
            target_text = f"{ans}" + " {sep_token}"

            examples.append(
                {
                    "source_text": input_text,
                    "target_text": target_text,
                    "task": "answer_ext",
                }
            )

        return examples

    def _generate_examples(self, filepath: str) -> Iterable:
        """
        This function returns the examples in the raw (text) form.
        """
        logging.info(f"Generating examples from {filepath}")
        count = 0

        with open(filepath) as f:
            articles = json.load(f)

        for article in articles["data"]:
            context = article["context"].strip()
            question = article["question"].strip()
            answers = article["answers"]

            # Generate the examples for answer extraction task.
            if "answer_ext" in self.config.sub_task:
                answer_ext_examples = self.process_answer_extraction(article)
                for answer_ext_example in answer_ext_examples:
                    yield count, answer_ext_example
                    count += 1

            # Generate the examples for QA, QG task.
            for task in self.config.sub_task:
                for start_idx, answer_text in zip(
                    answers["answer_start"], answers["text"]
                ):
                    answer = {
                        "answer_start": [start_idx],
                        "text": [answer_text],
                    }
                    if task == "qg":
                        yield count, self.process_qg_text(context, question, answer)
                        count += 1

                    if task == "qa":
                        yield count, self.process_qa_text(context, question, answer)
                        count += 1
