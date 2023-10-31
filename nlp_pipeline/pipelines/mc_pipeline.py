import gc
import re
from itertools import accumulate, chain
from typing import Generator, List, Optional, Union

from transformers import PreTrainedModel, PreTrainedTokenizer

from questgen.pipelines.modules.complement_distractors.generate_distractors import (
    DistractorComplement,
)
from questgen.pipelines.modules.complement_distractors.mc_generator import MCGenerator
from questgen.pipelines.modules.postprocess import __mapping__ as postprocess
from questgen.pipelines.modules.preprocess import __mapping__ as preprocess
from questgen.utils.constants import WORKERS
from questgen.utils.file_utils import logger
from questgen.utils.utils import get_progress, multiprocess


class MCPipeline(MCGenerator):
    """
    Initialize a QG pipeline for question generation task.
    """

    def __init__(
        self,
        use_multiprocess: bool,
        num_options: int,
        fillmask_model_path: str = None,
        model: Optional[PreTrainedModel] = None,
        tokenizer: Optional[Union[str, PreTrainedTokenizer]] = None,
        **config,
    ):
        super().__init__(model, tokenizer, **config)
        self.use_multiprocess = use_multiprocess
        self.num_options = num_options
        self.mapping_answer = {"A": 0, "B": 1, "C": 2, "D": 3}
        self.fillmask_model_path = fillmask_model_path

        if fillmask_model_path is not None:
            self.init_distractors = DistractorComplement(fillmask_model_path)

    def __call__(self, examples: List[dict], lang="vi"):
        examples = self.preprocess_input_mc(examples)
        distractors = self.generate(examples)
        output = self.postprocess_output_mc(
            examples=examples, distractors=distractors, lang=lang
        )
        del distractors, examples
        gc.collect()
        return output

    def preprocess_input_mc(self, examples):
        examples = self._multiprocess_or_mappingfunc(
            examples, self._convert_multitask_to_mc_format
        )
        return examples

    def postprocess_output_mc(self, examples, distractors, lang: str = "vi"):
        self.lang = lang

        examples = list(self._split_examples(examples, distractors))
        contexts = [example[0] for example in examples]
        questions = [example[1] for example in examples]
        mc_answers = [example[2] for example in examples]
        org_contexts = [example[3] for example in examples]
        key_info = [example[4] for example in examples]
        mc_answers = self._multiprocess_or_mappingfunc(
            mc_answers, self._postprocess_answers
        )
        if self.fillmask_model_path is not None:
            mc_answers = map(self._complement_distractor, mc_answers)

        flat_answers = self._multiprocess_or_mappingfunc(
            mc_answers, self._format_answers
        )

        assert len(contexts) == len(questions) == len(flat_answers)
        output = [
            {
                "org_context:": org_context,
                "context": context,
                "question": question,
                "answers": answers,
                "key_info": key.strip(),
            }
            for org_context, context, question, answers, key in zip(
                org_contexts, contexts, questions, flat_answers, key_info[0]
            )
        ]

        return output

    def _multiprocess_or_mappingfunc(self, examples: list, function):
        if self.use_multiprocess:
            examples = multiprocess(
                function, examples, workers=WORKERS, desc="...using multiprocess..."
            )
        else:
            examples = list(
                map(function, get_progress(examples, desc="...mapping function..."))
            )
        return examples

    def _convert_multitask_to_mc_format(self, examples: List[dict]):
        """This function aims to convert input's multitask format
        to input's multiplechoice format.
        """
        examples["context"] = examples["org_context"]
        examples["question"] = [examples["question"]]
        examples["options"] = [examples["answer"]]
        examples["answer"] = "A"
        return examples

    def _format_answers(self, mc_answers: List[str]):
        mc_answers = self._postprocess_answers(mc_answers)
        mc_answers = postprocess["remove_duplicate_answer"](mc_answers)
        mc_answers = postprocess["rank_answer_withmatching"](mc_answers)
        mc_answers = [
            f"{answer[0].upper()}" + f"{answer[1:]}" for answer in mc_answers
        ]  # capitalize the first character
        mc_answers = mc_answers[: self.num_options]
        return mc_answers

    def _postprocess_answers(self, mc_answers: List[str]):
        mc_answers = postprocess["strip_space"](mc_answers)
        mc_answers = postprocess["remove_duplicate_answer"](mc_answers)
        if self.lang == "vi":
            mc_answers = postprocess["remove_short_answer"](mc_answers)
            mc_answers = postprocess["remove_hl_character"](mc_answers)
        elif self.lang == "eng":
            mc_answers = postprocess["remove_short_answer"](mc_answers)
            mc_answers = postprocess["remove_hl_character"](mc_answers)
            mc_answers = postprocess["remove_noise"](mc_answers)
            mc_answers = postprocess["remove_duplicate_answers_english"](mc_answers)
            mc_answers = postprocess["remove_dots_answers"](mc_answers)
            # mc_answers = postprocess["add_empty_answer"](mc_answers)
        return mc_answers

    def _complement_distractor(self, mc_answers: Union[List, Generator]):
        """
        This function complements distrators
        in case not enough distractors generated.

        Args:
            mc_answers (Union[List, Generator]): multiplechoice-anwers

        Yields:
            generator: mc_answes[correct_answer, distractor[1-3]
        """
        substract = self.num_options - len(mc_answers)
        correct_answer = mc_answers[0]

        num_pattern = self._find_pattern(mc_answers)
        if num_pattern is not None:
            distractors = postprocess["numerical_pattern"](
                num_pattern, correct_answer, substract
            )
            [mc_answers.insert(-1, f"{distractor}") for distractor in distractors]
        else:
            if substract > 0:
                try:
                    distractors = self.init_distractors.complement(
                        sentence=correct_answer, num_masks=2, top_result=None
                    )
                    [
                        mc_answers.insert(-1, f"{distractor}")
                        for distractor in distractors
                    ]
                except Exception as e:
                    logger.info(e)
        return mc_answers

    def map_answer(self, examples):
        return examples["options"][self.mapping_answer[examples["answer"]]]

    def _split_examples(self, examples: list, mc_answers: list) -> Generator:
        correct_answers = [self.map_answer(example) for example in examples]

        contexts = [example["context"] for example in examples]
        questions = [example["question"] for example in examples]
        org_contexts = [example["org_context"] for example in examples]
        key_info = [example["key_info"] for example in examples]
        num_split = [self.num_sequences] * (len(mc_answers) // (self.num_sequences))
        mc_answers = self.split_lists(mc_answers, num_split)
        for idc, sub_answer in enumerate(mc_answers):
            try:
                sub_answer = set(list(chain(*sub_answer)))
                sub_answer = [
                    answer for answer in sub_answer if answer != correct_answers[idc]
                ]
                sub_answer.insert(0, correct_answers[idc])
                yield (
                    contexts[idc],
                    questions[idc],
                    sub_answer,
                    org_contexts[idc],
                    key_info,
                )

            except Exception as e:
                logger.info(e)

    def _find_pattern(self, mc_answers: Union[List, Generator]) -> str:
        """
        This function finds the pattern of mc_answers
        and divide into 2 cases:
            + Numerical pattern (contains digits).
            + Other patterns.
        """
        correct_answer = mc_answers[0]
        num_pattern = re.findall("[0-9]+", correct_answer)
        if len(num_pattern) > 0:
            return num_pattern
        else:
            return None

    def _prepare_inputs_for_mc(self, examples: dict):
        options = examples["options"]
        correct_answer = options[self.mapping_answer[examples["answer"]]]

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

    @staticmethod
    def split_lists(inputs: List, lengths_to_split: List[int]) -> List[str]:
        output = [
            inputs[x - y : x]
            for x, y in zip(accumulate(lengths_to_split), lengths_to_split)
        ]
        return output
