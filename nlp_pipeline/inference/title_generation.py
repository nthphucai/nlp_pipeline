import json
from random import randrange
from typing import Dict, Optional

import nltk
from transformers import T5ForConditionalGeneration, T5Tokenizer


class TitleGenerator:
    def __init__(self, model_path: str, config: Optional[Dict] = None):
        self.model = T5ForConditionalGeneration.from_pretrained(model_path)
        self.tokenizer = T5Tokenizer.from_pretrained(model_path)
        self.config = config
        if self.config is None:
            self.config = {
                "max_input_length": 768,
                "min_output_length": 5,
                "max_output_length": 24,
                "device": "cuda",
                "num_beams": 2,
                "do_sample": False,
                "truncation": True,
            }

    def generate(self, context: str) -> str:
        """
        Generate title from given context.

        Args:
            context (str): Context to generate title.

        Returns:
            Title for context.
        """
        inputs = "summarize: " + context
        inputs = self.tokenizer(
            inputs,
            max_length=self.config["max_input_length"],
            truncation=self.config["truncation"],
            return_tensors="pt",
        ).input_ids.to(self.config["device"])
        self.model.to(self.config["device"])
        output = self.model.generate(
            inputs,
            num_beams=self.config["num_beams"],
            do_sample=self.config["do_sample"],
            min_length=self.config["min_output_length"],
            max_length=self.config["max_output_length"],
        )
        decoded_output = self.tokenizer.batch_decode(output, skip_special_tokens=True)[
            0
        ]
        try:
            predicted_title = nltk.sent_tokenize(decoded_output.strip())[0]
        except LookupError:
            nltk.download("punkt")
            predicted_title = nltk.sent_tokenize(decoded_output.strip())[0]
        return predicted_title

    @staticmethod
    def get_question(data_path: str):
        """
        Get a question of 'Choosing title' task.
        Args:
            data_path (str): path to json file contain questions of 'Choosing title' task.

        Returns:
            A question of 'Choosing title' task.
        """
        question = json.load(open(data_path, "r", encoding="utf-8"))["data"]
        return question[randrange(len(question) - 1)]
