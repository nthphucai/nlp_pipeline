from typing import Tuple

import torch
from torch.utils.data import DataLoader as dloader

from questgen.ranking.qa_utils import highlight_words, prepare_hl


class DataProcessor:
    def __init__(self, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def process(self, dataset, bz, mode="eval"):
        self.mode = mode
        dataset = list(map(self._prepare_data_for_ranking, dataset))
        encodings = list(map(self._convert_to_features, dataset))
        encodings_loader = dloader(
            encodings, batch_size=bz, collate_fn=self._generate_batch
        )

        return encodings_loader

    def _prepare_data_for_ranking(self, example: dict):
        context = example["context"]
        question = example["question"]
        answers = example["answer"]

        # context = self._extract_pos_answers(context, answers)
        hl_question = prepare_hl(keys=question, stopwords=None)
        hl_answer = prepare_hl(keys=answers, stopwords=None)
        hl_question.extend(hl_answer)
        context = highlight_words(context, hl_question)

        pair = (
            question + "<sep>" + answers + "<sep>" + context + self.tokenizer.eos_token
        )
        return pair

    def _extract_pos_answers(self, context: str, answer: str) -> Tuple[int, int, str]:
        pos = context.lower().find(answer.lower())
        context = context[0:pos] + "<hl>" + answer + "<hl>" + context[len(answer) :]
        return context

    def _convert_to_features(self, example_batch):
        encodings = self.tokenizer.encode_plus(
            example_batch,
            max_length=self.max_length,
            padding="longest",
            add_special_tokens=True,
            pad_to_max_length=True,
            truncation=True,
        )
        encodings = {
            "input_ids": torch.tensor(encodings["input_ids"]),
            "attention_mask": torch.tensor(encodings["attention_mask"]),
        }
        return encodings

    def _generate_batch(self, example_batch: list) -> dict:
        from torch.nn.utils.rnn import pad_sequence

        input_ids = pad_sequence(
            [example["input_ids"] for example in example_batch], batch_first=True
        )
        attention_mask = pad_sequence(
            [example["attention_mask"] for example in example_batch], batch_first=True
        )
        batch = {"input_ids": input_ids, "attention_mask": attention_mask}
        return batch
