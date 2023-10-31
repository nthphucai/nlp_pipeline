import itertools
import re
from itertools import accumulate, chain
from typing import Dict, Iterable, List, Tuple, Union

import numpy as np
import pandas as pd
import torch
from nltk import sent_tokenize
from torch.utils.data import DataLoader as load_batch
from transformers import PreTrainedModel, PreTrainedTokenizer

from fib_questgen.models.qa_generator import QAGenerator
from questgen.pipelines.modules.preprocess import __mapping__ as preprocess
from questgen.ranking.processor import DataProcessor
from questgen.ranking.qa_utils import load_sklearn_model
from questgen.utils.constants import WORKERS
from questgen.utils.model_utils import extract_features
from questgen.utils.utils import get_progress


class QAPipeline:
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        max_source_length: int = 512,
        max_target_length: int = 64,
        num_beams: int = 4,
        length_penalty: float = 1.0,
        early_stopping: bool = True,
        qg_format: str = "highlight",
        use_cuda: bool = True,
        batch_size: int = 8,
        num_answer_sequences: int = 1,
        num_question_sequences: int = 1,
        do_sample_answers: bool = False,
        do_sample_question: bool = False,
        top_p: float = 0.95,
        fast_inference: bool = True,
        model_type: str = "t5",
    ):
        self.model = model
        self.tokenizer = tokenizer

        self.qg_format = qg_format
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.num_beams = num_beams
        self.batch_size = batch_size
        self.length_penalty = length_penalty
        self.early_stopping = early_stopping

        self.n_answer_sequences = num_answer_sequences
        self.n_question_sequences = num_question_sequences
        self.do_sample_answers = do_sample_answers
        self.do_sample_question = do_sample_question
        self.top_p = top_p

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and use_cuda else "cpu"
        )
        self.model.to(self.device)

        if not fast_inference:
            assert self.model.__class__.__name__ in [
                "T5ForConditionalGeneration",
                "BartForConditionalGeneration",
                "ORTModelForSeq2SeqLM",
            ]

            if (
                "T5ForConditionalGeneration"
                or "ORTModelForSeq2SeqLM" in self.model.__class__.__name__
            ):
                self.model_type = "t5"
            else:
                self.model_type = "bart"
            if "ORTModelForSeq2SeqLM" not in self.model.__class__.__name__:
                self.model.eval()
        else:
            self.model_type = model_type

    @staticmethod
    def detach_hl_sentence(text):
        sentence_list = [t.strip() for t in text.split(". ")]
        main_info = ""
        in_series = False
        for sent in sentence_list:
            num_hl = sent.count("<hl>")
            if num_hl == 2:
                main_info = sent
                break
            elif sent.count("<hl>") == 1:
                main_info += sent
                in_series = not in_series
            elif in_series:
                main_info += f". {sent}"

        main_info = main_info.replace("generate question: ", "")
        main_info = main_info.replace("<hl>", "")
        main_info = re.sub(r"\s+", " ", main_info)
        main_info = QAGenerator.normalize_paragraph(main_info)
        return main_info

    def generate_batch_qa_pairs(self, examples: Union[dict, List[dict]]):
        examples = list(self._prepare_inputs_for_batch_questions(examples))
        lengths = [self.n_question_sequences * example[2] for example in examples]
        inputs = [example[3] for example in examples]
        origin_contexts = [example[4] for example in examples]

        answers = list(chain(*[example[0] for example in examples]))
        qg_examples = chain(*[example[1] for example in examples])

        questions = []
        for answer in get_progress(
            load_batch(answers, batch_size=self.batch_size, num_workers=WORKERS),
            desc="...generate-batch-questions...",
        ):
            question = self._generate_questions(answer)
            questions.extend(question)

        questions = list(chain(*questions))
        answers = np.repeat(
            [qg_example["answer"] for qg_example in qg_examples],
            self.n_question_sequences,
        )

        assert len(questions) == len(answers), f"{len(questions)} {len(answers)}"
        outputs = [
            {"question": question, "answer": answer}
            for question, answer in zip(questions, answers)
        ]

        return (outputs, lengths, inputs, origin_contexts)

    def _generate_one_keyinfo(self, answer: str, context: str):
        sentences = preprocess["sent_segment"](context)
        sentences = [sent for sent in sentences if sent is not None]
        hl_sentence = [sent for sent in sentences if answer in sent]
        return hl_sentence

    def generate_batch_answers_for_qa(self, examples):
        answers = list(map(self._prepare_inputs_for_qa, examples))
        questions = [example["question"] for example in examples]
        questions = [[question] * self.n_answer_sequences for question in questions]
        origin_contexts = [example["org_context"] for example in examples]
        lengths = [self.n_answer_sequences] * len(origin_contexts)

        answers_lst = []
        for answer in get_progress(
            load_batch(answers, batch_size=self.batch_size, num_workers=WORKERS),
            desc="...question-batch-answer-for-qa...",
        ):
            inputs = self._tokenize(
                answer, padding=True, truncation=True, max_length=self.max_source_length
            )
            outs = self.model.generate(
                input_ids=inputs["input_ids"].to(self.device),
                attention_mask=inputs["attention_mask"].to(self.device),
                max_length=self.max_source_length,
                num_return_sequences=self.n_answer_sequences,
                do_sample=self.do_sample_answers,
            )
            answers_lst.append(
                self.tokenizer.batch_decode(outs, skip_special_tokens=True)
            )

        answers_lst, questions = [
            list(chain(*item)) for item in (answers_lst, questions)
        ]
        assert len(answers_lst) == len(questions)

        outputs = [
            {"question": question, "answer": answer}
            for question, answer in zip(questions, answers_lst)
        ]
        return outputs, lengths, examples, origin_contexts

    def format_rankingQA(self, inputs: List[dict], qa_pairs: List[dict]):
        context = inputs["context"]
        result = self._rank_qa_pairs(context=context, qa_pairs=qa_pairs)

        output = dict()
        output["prob"] = [result[idc][0] for idc in range(len(result))]
        output["question"] = [result[idc][1] for idc in range(len(result))]
        output["answer"] = [result[idc][2] for idc in range(len(result))]

        qa_pair = [
            {"question": question, "answer": answer}
            for question, answer in zip(output["question"], output["answer"])
        ]

        for idc in range(len(qa_pair)):
            qa_pair[idc]["prob"] = output[idc]["prob"]

        result = pd.DataFrame(qa_pair).drop_duplicates(subset=["answer"])
        result.to_dict(orient="recored")
        return result

    def _prepare_inputs_for_qa(self, example: dict):
        question, context = example["question"], example["context"]
        source_text = f"question: {question} context: {context}"
        if self.model_type == "t5":
            source_text = source_text + " </s>"
        return source_text

    def _prepare_inputs_for_answer_extraction(self, text: str):
        try:
            sentences = sent_tokenize(text)
        except LookupError:
            import nltk

            nltk.download("punkt")
            sentences = sent_tokenize(text)

        inputs = []
        for i in range(len(sentences)):
            source_text = "extract answers:"
            for j, sentence in enumerate(sentences):
                if i == j:
                    sentence = f"<hl> {sentence} <hl>"
                source_text = f"{source_text} {sentence}"
                source_text = source_text.strip()

            source_text = source_text + " </s>"
            inputs.append(source_text)

        return sentences, inputs

    def _tokenize(
        self,
        inputs,
        padding=True,
        truncation=True,
        add_special_tokens=True,
        max_length=512,
    ):
        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token = self.tokenizer.eos_token

        inputs = self.tokenizer.batch_encode_plus(
            inputs,
            max_length=max_length,
            add_special_tokens=add_special_tokens,
            truncation=truncation,
            padding="longest" if padding else False,
            pad_to_max_length=padding,
            return_tensors="pt",
        )
        return inputs

    def _extract_batch_answers_for_qg(
        self, inputs: List[dict]
    ) -> Tuple[List[dict], tuple]:
        org_context = [inputs[idc]["org_context"] for idc in range(len(inputs))]
        context = [inputs[idc]["context"] for idc in range(len(inputs))]

        examples = list(map(self._prepare_inputs_for_answer_extraction, context))
        answers = list(chain(*[example[1] for example in examples]))
        sentences = [example[0] for example in examples]
        lengths = [len(example[1]) for example in examples]

        answers_lst = []
        for answers in get_progress(
            load_batch(answers, batch_size=self.batch_size, num_workers=WORKERS),
            desc="...extract-batch-answers-for-qg...",
        ):
            answers = self._tokenize(
                inputs=answers,
                padding=True,
                truncation=True,
                max_length=self.max_source_length,
            )

            outs = self.model.generate(
                input_ids=answers["input_ids"].to(self.device),
                attention_mask=answers["attention_mask"].to(self.device),
                max_length=self.max_target_length,
                num_return_sequences=self.n_answer_sequences,
                do_sample=self.do_sample_answers,
            )

            decode_answers = [
                self.tokenizer.batch_decode(outs, skip_special_tokens=True)
            ]
            decode_answers = [ans.split("<sep>")[0] for ans in decode_answers[0]]
            answers = self.split_lists(
                decode_answers,
                [self.n_answer_sequences] * answers["input_ids"].shape[0],
            )
            answers_lst.extend(answers)

        answers = self.split_lists(answers_lst, lengths)
        pairs = tuple(zip(sentences, answers))
        assert len(inputs) == len(
            pairs
        ), "context length must be equal (sent, ans) length"
        return inputs, pairs, org_context

    def _extract_pos_answers(context: str, answer: str) -> Tuple[int, int, str]:
        match = re.search(answer, context)
        start_pos = match.start()
        end_pos = match.end()
        return start_pos, end_pos, answer

    def _prepare_inputs_for_qg_from_answers_hl(
        self, context: Union[str, List[str]], answers: Union[str, List[str]]
    ) -> List[Dict[str, str]]:
        examples = []
        if isinstance(context, str):
            start_pos, end_pos, _ = self._extract_pos_answers(context, answers)

            answer_text = answers.strip()
            source_text = (
                f"{context[:start_pos]}<hl> {answer_text} <hl>{context[end_pos:]}"
            )
            source_text = f"generate question: {source_text}"

            source_text = source_text + " </s>"

            examples.append({"answer": answer_text, "source_text": source_text})
        else:
            for i, answer in enumerate(answers):
                if len(answer) == 0:
                    continue
                for answer_text in answer:
                    try:
                        sentence = context[i]
                        sentences_copy = context[:]

                        answer_text = answer_text.strip()
                        answer_start_idx = sentence.index(answer_text)

                        sentence = f"{sentence[:answer_start_idx]} <hl> {answer_text} <hl> {sentence[answer_start_idx + len(answer_text):]}"
                        sentences_copy[i] = sentence

                        source_text = " ".join(sentences_copy)
                        source_text = f"generate question: {source_text}"

                        source_text = source_text + " </s>"

                        examples.append(
                            {"answer": answer_text, "source_text": source_text}
                        )
                    except Exception:
                        continue

        return examples

    def _generate_questions(self, inputs: List[str]) -> List[Dict[str, str]]:
        inputs = self._tokenize(
            inputs, padding=True, truncation=True, max_length=self.max_source_length
        )

        outs = self.model.generate(
            input_ids=inputs["input_ids"].to(self.device),
            attention_mask=inputs["attention_mask"].to(self.device),
            max_length=self.max_target_length,
            num_beams=self.num_beams,
            length_penalty=self.length_penalty,
            num_return_sequences=self.n_question_sequences,
            do_sample=self.do_sample_question,
            top_p=self.top_p,
            early_stopping=self.early_stopping,
        )

        questions = [self.tokenizer.batch_decode(outs, skip_special_tokens=True)]
        return questions

    def _prepare_inputs_for_batch_questions(self, examples: List[dict]) -> Iterable:
        context, pairs, org_context = self._extract_batch_answers_for_qg(examples)
        for idc, (inputs, answers) in enumerate(pairs):
            qg_examples = self._prepare_inputs_for_qg_from_answers_hl(inputs, answers)
            qg_inputs = [example["source_text"] for example in qg_examples]
            yield qg_inputs, qg_examples, len(qg_inputs), context[idc], org_context[idc]

    @staticmethod
    def split_lists(inputs: List[str], lengths_to_split: List[int]) -> List[str]:
        # split a list into sublist of given lengths
        output = [
            inputs[x - y : x]
            for x, y in zip(accumulate(lengths_to_split), lengths_to_split)
        ]
        return output

    def _rank_qa_pairs(
        self,
        context: str,
        qa_pairs: List[dict],
        gaussian_mixture_model: str = "data/qa_ranking/gaussian_mixture.dump",
        isotonic_regressor_model: str = "data/qa_ranking/isotonic_regressor.dump",
    ):
        qa_pairs = list(itertools.chain(*[qa_pairs]))
        for idc in range(len(qa_pairs)):
            qa_pairs[idc]["context"] = context

        processor = DataProcessor(tokenizer=self.tokenizer, max_length=256)
        data_loader = processor.process(qa_pairs, bz=4, mode="test")
        features = extract_features(
            data_loader, self.model, verbose=True, device="cuda"
        )
        assert (features.shape[0]) == len(
            qa_pairs
        ), f"the feature shape {features.shape[0]} must be equal data shape {len(qa_pairs)}"

        # load gaussian_mixture model, isotonic_regressor model
        gaussian_mixture = load_sklearn_model(gaussian_mixture_model)
        isotonic_regressor = load_sklearn_model(isotonic_regressor_model)
        log_probs = gaussian_mixture.score_samples(features)
        probs = isotonic_regressor.predict(log_probs)
        assert len(probs) == len(qa_pairs), f"{len(probs)} must be {len(qa_pairs)}"

        results = []
        for idc in range(len(qa_pairs)):
            result = (
                np.round(probs[idc], 3),
                qa_pairs[idc]["question"],
                qa_pairs[idc]["answer"],
            )
            results.append(result)

        results = [
            value[1]
            for value in sorted(
                enumerate(results), key=lambda item: item[1][0], reverse=True
            )
        ]
