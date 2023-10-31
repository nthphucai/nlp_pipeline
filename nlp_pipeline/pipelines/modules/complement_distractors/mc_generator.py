from typing import Dict, List

import torch
from torch.utils.data import DataLoader as load_batch
from transformers import PreTrainedModel, PreTrainedTokenizer

from questgen.pipelines.modules.preprocess import __mapping__ as preprocess
from questgen.utils.constants import WORKERS
from questgen.utils.utils import get_progress


class MCGenerator:
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        batch_size: int = 16,
        use_cuda: bool = True,
        max_source_length: int = 512,
        max_target_length: int = 64,
        num_beams: int = 2,
        num_sequences: int = 1,
        length_penalty: float = 1.0,
        repetition_penalty: int = 2.5,
        do_sample: bool = False,
        num_beam_groups: int = 2,
        diversity_penalty: float = 1.0,
        top_p: float = 1.0,
        early_stopping: bool = True,
    ):
        self.model = model
        self.tokenizer = tokenizer

        self.batch_size = batch_size
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.num_beams = num_beams
        self.length_penalty = length_penalty
        self.num_sequences = num_sequences
        self.repetition_penalty = repetition_penalty
        self.do_sample = do_sample
        self.num_beam_groups = num_beam_groups
        self.diversity_penalty = diversity_penalty
        self.top_p = top_p
        self.early_stopping = early_stopping

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and use_cuda else "cpu"
        )
        self.model.to(self.device)
        self.model.eval()

        assert self.model.__class__.__name__ in [
            "T5ForConditionalGeneration",
            "BartForConditionalGeneration",
        ]

        if "T5ForConditionalGeneration" in self.model.__class__.__name__:
            self.model_type = "t5"
        else:
            self.model_type = "bart"

        if self.model_type == "t5":
            self.sep_token = "<sep>"
        elif self.model_type == "bart":
            self.sep_token = "<sep>"
        else:
            self.sep_token = "[SEP]"

        self.mapping_answer = {"A": 0, "B": 1, "C": 2, "D": 3}

    def generate(self, examples: List[dict]):
        mc_example = map(self._prepare_inputs_for_mc, examples)
        source_text = [example["source_text"] for example in mc_example]
        distractors = []
        for answer in get_progress(
            load_batch(source_text, batch_size=self.batch_size, num_workers=WORKERS),
            desc="...generate-distractor...",
        ):
            distractor = self.generate_distractors(answer)
            distractors.extend(distractor)
        return distractors

    def generate_distractors(self, inputs: List[str]) -> List[Dict[str, str]]:
        """This function aims to generate distractors from correct answers.

        Args:
            inputs (List[str]): List of correct answers.

        Returns:
            List[Dict[str, str]]: List of distractors corresponding correct answers.
        """
        inputs = self._tokenize(
            inputs, padding=True, truncation=True, max_length=self.max_source_length
        )
        # diverse beam-search decoding by calling group_beam_search()
        # if num_beams > 1 and num_beam_groups > 1.
        outs = self.model.generate(
            input_ids=inputs["input_ids"].to(self.device),
            attention_mask=inputs["attention_mask"].to(self.device),
            max_length=self.max_target_length,
            num_beams=self.num_beams,
            length_penalty=self.length_penalty,
            repetition_penalty=self.repetition_penalty,
            num_return_sequences=self.num_sequences,
            num_beam_groups=self.num_beam_groups,
            diversity_penalty=self.diversity_penalty,
            top_p=self.top_p,
            do_sample=self.do_sample,
            early_stopping=self.early_stopping,
        )
        decode_mc = [self.tokenizer.batch_decode(outs, skip_special_tokens=True)]
        mc_answers = [item.split("<sep>") for item in decode_mc[0]]
        return mc_answers

    def _tokenize(
        self,
        inputs,
        padding=True,
        truncation=True,
        add_special_tokens=True,
        max_length=512,
    ):
        inputs = self.tokenizer.batch_encode_plus(
            inputs,
            max_length=max_length,
            add_special_tokens=add_special_tokens,
            truncation=truncation,
            padding="max_length" if padding else False,
            pad_to_max_length=padding,
            return_tensors="pt",
        )
        return inputs

    def prepare_inputs_for_mc(self, examples: dict):
        options = examples["options"]
        correct_answer = options[self.mapping_answer[examples["answers"]]]

        question = examples["question"][0]
        context = examples["context"]
        context = preprocess["viquad_noise"](context)

        source_text = (
            f"{correct_answer} {{sep_token}} {question} {{sep_token}} {context}"
        )
        if self.model_type == "t5":
            source_text = source_text + " </s>"

        source_text = source_text.replace("{sep_token}", self.sep_token)
        return {"source_text": source_text}
