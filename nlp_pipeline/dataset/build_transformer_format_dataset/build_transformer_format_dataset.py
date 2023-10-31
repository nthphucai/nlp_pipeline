import json
import logging
from typing import Iterable

import datasets
import nltk

from constant import INSTRUCTION
from nlp_pipeline.modules.gen_llm.prompt_template import formatted_prompt

nltk.download("punkt")

_DESCRIPTION = """\Introduction about dataset.
"""

_CITATION = """\
"""

class LLMGenConfig(datasets.BuilderConfig):

    def __init__(self, llm_architect:str="encoder-decoder", **kwargs):
        """BuilderConfig for Datasets.
        Args:
            **kwargs: keyword arguments forwarded to super.
        """
        super(LLMGenConfig, self).__init__(**kwargs)
        self.llm_architect = llm_architect


class Text2TextGeneration(datasets.GeneratorBasedBuilder):
    """Datasets Information"""

    BUILDER_CONFIGS = [
        LLMGenConfig(
            name="default",
            version=datasets.Version(
                "1.1.0", "New split API (https://tensorflow.org/datasets/splits)"
            ),
            description="Plain text",
            llm_architect=LLMGenConfig().llm_architect,
        )
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
            homepage="",
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

    def process_text_seg2segLM(self, requirement: str, testcase: dict) -> dict:
        gen_input = f"requirement: {requirement}"
        gen_target = "<sep>".join([f"{k} {testcase[k]}" for k in testcase])

        examples = {
            "source_text": gen_input,
            "target_text": gen_target,
            "task": "seg2seg-lm",
        }

        return examples
    
 
    def process_prompt_casualLM(self, inpt:str, response:str):
        prompt = formatted_prompt.format(INSTRUCTION, inpt, response)
        return {
            "source_text": prompt,
            "target_text": "",
            "task": "casual-lm",
        }

    def _generate_examples(self, filepath: str) -> Iterable:
        """
        This function returns the examples in the raw (text) form.
        """
        logging.info(f"Generating examples from {filepath}")
        with open(filepath) as f:
            data = json.load(f)

        input_key = "Requirement Text" #"dialogue"
        output_key = "Test Cases" #"summary"

        for cnt, item in enumerate(data["data"]):
        #for cnt, item in enumerate(data):
            requirement = item[input_key]
            testcase = item[output_key]

            if self.config.llm_architect == "decoder-only":
                yield cnt, self.process_prompt_casualLM(requirement, testcase)
            elif self.config.llm_architect == "encoder-decoder":
                yield cnt, self.process_text_seg2segLM(requirement, testcase)
            else:
                raise NotImplementedError("Only support llm architecture `decoder-only` and `encoder-decoder`")
 
        