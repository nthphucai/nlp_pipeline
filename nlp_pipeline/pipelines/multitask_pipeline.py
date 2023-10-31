import gc
from itertools import chain
from typing import List, Optional, Tuple, Union

import pandas as pd

from questgen.pipelines.modules.preprocess import __mapping__ as preprocess
from questgen.pipelines.modules.summarize.summary_text import (
    TextSummarize as summary_inputs,
)
from questgen.pipelines.qg_pipeline import QAPipeline
from questgen.utils.constants import WORKERS
from questgen.utils.utils import get_progress, multiprocess


class MultiTaskPipeline(QAPipeline):
    def __init__(self, use_multiprocess: bool = False, **kwargs):
        super().__init__(**kwargs)

        self.use_multiprocess = use_multiprocess

    def __call__(
        self,
        # task: str = "question-generation",
        task: str = "question-answering",
        examples: Union[dict, List[dict]] = None,
        use_summary: bool = True,
    ) -> List[dict]:
        """Given dictionary input of context (question-generation task)
        and return their corresponding question-answer pairs or
        given dictionary input of context-question pairs (question-answering task)
        and return their corresponding answers.

        Args:
            task (str, optional): The supported task. Defaults to "question-generation".
            examples (Union[dict, List[dict]], optional): The list of dictionaries. Defaults to None.
            use_summary (bool, optional): Whether to use summary for simple question task. Defaults to True.

        Returns:
            List[dict]: The list of dictionary outputs.

        Example:
        >>> examples = [{
            "context": "Vào tháng 4 năm 2010, tỷ lệ thất nghiệp chính thức là 9,9%, nhưng tỷ lệ thất nghiệp theo cách tính U6 của chính phủ (U-6 unemployment) là 17,1%. Trong thời kỳ giữa tháng 2 ..."}]

            [{
                "context": "Vào tháng 4 năm 2010, tỷ lệ thất nghiệp chính thức là 9,9%, nhưng tỷ lệ thất nghiệp theo cách tính U6 của chính phủ (U-6 unemployment) là 17,1%. Trong thời kỳ giữa tháng 2 ...",
                "org_context": "Vào tháng 4 năm 2010, tỷ lệ thất nghiệp chính thức là 9,9%, nhưng tỷ lệ thất nghiệp theo cách tính U6 của chính phủ (U-6 unemployment) là 17,1%. Trong thời kỳ giữa tháng 2 ...",
                "key_info": ["Vào tháng 4 năm 2010, tỷ lệ thất nghiệp chính thức là 9,9%, nhưng tỷ lệ thất nghiệp theo cách tính U6 của chính phủ (U-6 unemployment) là 17,1%"],
                "question": "Tỉ lệ thất nghiệp theo tính U6 là bao nhiêu?",
                "answer": "17,1%"
            }]

        """
        self.use_summary = use_summary
        examples = [examples] if type(examples) is not list else examples
        assert isinstance(examples[0], dict), "The example must be a dict"
        assert task in (
            "question-generation",
            "question-answering",
        ), "only supoirt question-generation or question-answering task"

        examples = self.preprocess_input_multitask(examples)
        if task == "question-generation":
            outputs = self.generate_batch_qa_pairs(examples)

        elif task == "question-answering":
            outputs = self.generate_batch_answers_for_qa(examples)

        qa_pairs = self.postprocess_output_multitask(outputs)
        information = self._generate_key_info(qa_pairs)
        [
            qa_pairs[idc].update({"key_info": information[idc]})
            for idc in range(len(information))
        ]

        del outputs, examples
        gc.collect()
        return qa_pairs

    def _generate_key_info(self, result: List[dict]) -> List[dict]:
        """Given the list of dictionary outputs and extract key_info where key_info
        is the sentence contains correct answer.

        Args:
            result (List[dict]): The list of dictionary input.

        Returns:
            List[dict]: The sentence contains correct answer.
        """
        anwers = [r["answer"] for r in result]
        context = [r["org_context"] for r in result]
        information = list(map(self._generate_one_keyinfo, anwers, context))
        return information

    def preprocess_input_multitask(self, examples):
        examples = self._multiprocess_or_mappingfunc(
            examples, self._preprocess_one_example
        )
        return examples

    def _multiprocess_or_mappingfunc(self, examples, function):
        """
        Whether to use multiprocess or mapping function,
        aims to test the performance.
        """
        if self.use_multiprocess:
            examples = multiprocess(
                function,
                examples,
                workers=WORKERS,
                desc="...preprocess inputs using multiprocess...",
            )
        else:
            examples = list(
                map(
                    function,
                    get_progress(
                        examples, desc="...preprocess inputs using mapping func..."
                    ),
                )
            )
        return examples

    def preprocess_one_gcn_example(self, example: dict):
        triples = example["triples"]

        current_kg = pd.DataFrame(triples, columns=["source", "relation", "target"])
        current_kg = current_kg.drop_duplicates(
            subset=["source", "target"], keep="first"
        )

        columns = ("source", "target", "relation")
        subject_token, target_token, rel_token = [
            pd.Series(current_kg[col]) for col in columns
        ]

        return {
            "source_subject": subject_token,
            "source_tgt": target_token,
            "source_rel": rel_token,
            "task": "gcn",
        }

    def _preprocess_one_example(self, example: dict):
        """Given the dictionary input and preprocess context before feeding
        to the model.

        Args:
            example (dict): The dictionary input having context as a key.

        Returns:
            dict: The dictionary having preprocessed context.

        Example:
        >>> example = {"context": ""Independent Women Part l" của nhóm nhạc Destiny's
        Child là đĩa đơn đứng đầu bảng lâu nhất năm 2000, với 11 tuần liên tiếp,[2]
        kéo dài thêm 4 tuần tại bảng xếp hạng năm 2001.}

        {"context": ""Independent Women Part l" của nhóm nhạc Destiny's Child
        là đĩa đơn đứng đầu bảng lâu nhất năm 2000, với 11 tuần liên tiếp,
        kéo dài thêm 4 tuần tại bảng xếp hạng năm 2001.}
        """
        if self.use_summary:
            example = self._summary_one_context(example)
        else:
            example["context"] = preprocess["viquad_noise"](example["context"])
            example["context"] = preprocess["hle"](example["context"])
            example["org_context"] = example["context"]
        return example

    def postprocess_output_multitask(self, outputs):
        """Postprocess the output, maping each dictionary with
        their correspoding context and drop the duplicates.

        Args:
            outputs (List[dict]): The list of dictionaries.

        Returns:
            List[dict]: The list of dictionaries.

        Example:
            >>> outputs = [{"question": question, "answer": answer}]
            >>> postprocess_output_multitask(outputs)
            [{"question": question, "answer": answer, "context": context, "org_context": org_context}]
        """
        qa_pairs = self._format_qa_pairs(*outputs, use_ranking=False, top_n=None)
        qa_pairs = multiprocess(
            self.mapping_qa_context,
            range(len(qa_pairs[0])),
            workers=WORKERS,
            result=qa_pairs,
        )
        qa_pairs = pd.DataFrame(chain(*qa_pairs))
        qa_pairs = qa_pairs.drop_duplicates(subset=["question", "answer"])
        qa_pairs = qa_pairs.to_dict("records")
        return qa_pairs

    def _summary_one_context(self, example: dict):
        example["org_context"] = example["context"]
        example["context"] = summary_inputs(example["context"], rate_cluster=0.6)()
        return example

    def _format_qa_pairs(
        self,
        outputs: List[dict],
        lengths: List[int],
        context: List[dict],
        org_contexts: List[dict],
        use_ranking: bool = False,
        top_n: Optional[int] = None,
    ) -> Union[Tuple[List[dict]], List[List[dict]]]:
        """This functions aims to ranking the results (default=False)
        and get top_n best result.

        Args:
            outputs (List[dict]): The list of dictionaries where keys are `question`,
            and the corresponding `answer`.
            lengths (List[int]): A list containing the numbers of context associated
            with the corresponding numbers of the answer.
            context (List[dict]): The summarized context from origin context.
            org_contexts (List[str]): The origin context.
            use_ranking (bool, optional): whether to rank result. Defaults to False.
            top_n (Optional[int], optional): top n ranking result. Defaults to None.

        Returns:
            Union[Tuple[List[dict]], List[List[dict]]]: The best ranking results.
        """
        outputs = self.split_lists(outputs, lengths)
        if use_ranking:
            outputs = list(map(self.format_rankingQA, context, outputs))

        if top_n is not None:
            outputs = [output[:top_n] for output in outputs]

        return outputs, context, org_contexts

    @staticmethod
    def mapping_qa_context(idc, result) -> List[dict]:
        """Given the list of dictionaries of question-answer pairs
        and mapping to their corresponding context.

        Args:
            idc (int): The index of list of dictionaries
            result (list[dict]): The list of dictionaries of qa pairs

        Returns:
            List[dict]: The list of dictionaries of qa pairs
            mapped to their context.
        """
        qa_result = result[0][idc]
        context_resut = result[1][idc]["context"]
        org_context_result = result[1][idc]["org_context"]
        for i in range(len(qa_result)):
            qa_result[i]["context"] = context_resut
            qa_result[i]["org_context"] = org_context_result
        return qa_result
