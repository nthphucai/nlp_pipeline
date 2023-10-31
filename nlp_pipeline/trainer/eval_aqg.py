import pathlib
import time
from dataclasses import dataclass, field
from os import PathLike
from typing import Dict, List, Optional, Union

import torch
from tqdm.auto import tqdm
from transformers import AutoModelForSeq2SeqLM, HfArgumentParser, T5Tokenizer

from nlgeval import compute_metrics as eval_automatic

from nlp_pipeline.dataset.build_transformer_format_dataset.data_collator import (
    Text2TextDataCollator,
)
from nlp_pipeline.utils import READ_FILE_FN
from nlp_pipeline.utils.file_utils import logger


@dataclass
class EvalArguments:
    task: str = field(
        metadata={"help": "Which task 'simple-question', 'multiple-choice'"}
    )
    model_name_or_path: str = field(
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        }
    )
    reference_path: Optional[str] = field(
        metadata={"help": "Whether save the ground truth reference text strings"}
    )
    valid_file_path: str = field(metadata={"help": "Path for cached valid dataset"})
    model_type: str = field(metadata={"help": "One of 't5', 'bart'"})
    tokenizer_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained tokenizer name or path if not the same as model_name"
        },
    )
    num_beams: Optional[int] = field(
        default=4, metadata={"help": "num_beams to use for decoding"}
    )
    max_decoding_length: Optional[int] = field(
        default=32, metadata={"help": "Maximum length for decoding"}
    )
    length_penalty: Optional[float] = field(
        default=1.0, metadata={"help": "length_penalty"}
    )
    repetition_penalty: Optional[float] = field(
        default=1.5, metadata={"help": "repetition_penalty"}
    )
    output_path: Optional[str] = field(
        default="questions.txt",
        metadata={"help": "Path to save the generated questions."},
    )
    eval_batch_size: int = field(
        default=8, metadata={"help": "Batch size for evaluation."}
    )
    model_type: str = field(
        default="t5", metadata={"help": "One of 't5', 't5-copy-enhance', 'bart'"}
    )
    device: str = field(default="cpu", metadata={"help": "Device"})


class QuestGenEvaluator:
    def __init__(
        self,
        task: str,
        model_name_or_path: str,
        valid_file_path: str,
        tokenizer_name_or_path: Optional[str] = None,
        model_type: str = "t5",
        num_beams: Optional[int] = 4,
        max_decoding_length: Optional[int] = 32,
        repetition_penalty: Optional[float] = 32,
        length_penalty: Optional[float] = 1.0,
        reference_path: Optional[str] = "references.txt",
        eval_batch_size: Optional[int] = 8,
        output_path: Optional[str] = None,
    ):
        self.task = task
        self.model_name_or_path = model_name_or_path
        self.valid_file_path = valid_file_path
        self.model_type = model_type
        self.num_beams = num_beams
        self.max_decoding_length = max_decoding_length
        self.repetition_penalty = repetition_penalty
        self.length_penalty = length_penalty
        self.reference_path = reference_path
        self.output_path = output_path
        self.eval_batch_size = eval_batch_size

        if tokenizer_name_or_path is None:
            tokenizer_name_or_path = model_name_or_path

        self.tokenizer_name_or_path = tokenizer_name_or_path

    def generate(self):
        args = EvalArguments(
            task=self.task,
            model_name_or_path=self.model_name_or_path,
            valid_file_path=self.valid_file_path,
            model_type=self.model_type,
            tokenizer_name_or_path=self.tokenizer_name_or_path,
            num_beams=self.num_beams,
            max_decoding_length=self.max_decoding_length,
            length_penalty=self.length_penalty,
            repetition_penalty=self.repetition_penalty,
            output_path=self.output_path,
            eval_batch_size=self.eval_batch_size,
            reference_path=self.reference_path,
        )
        generate(args=args)

    def compute_metrics(
        self,
        result_save_path: str,
        no_glove: bool = True,
        no_skipthoughts: bool = True,
        no_overlap: bool = False,
    ) -> Dict[str, Optional[float]]:
        """
        Evaluation for various unsupervised automated metrics for NLG (Natural Language Generation). It takes as
        input a hypothesis file, and one or more references files and outputs values of metrics. Rows across these
        files should correspond to the same example.

        Args:
            hypothesis: filepath of hypothesis file.
            references: list of references filepath.
            no_glove: option to using glove metric.
            no_skipthoughts: option to use skip_thought.
            no_overlap: option to use no_overlap.

        Returns:
        Dictionary contain various unsupervised automated metrics for NLG.
        """
        hypothesis_path = pathlib.Path(self.output_path)
        references = self.check_inputs_for_nlgeval(hypothesis_path, self.reference_path)
        eval_result = eval_automatic(
            hypothesis=hypothesis_path,
            references=references,
            no_glove=no_glove,
            no_skipthoughts=no_skipthoughts,
            no_overlap=no_overlap,
        )
        with open(result_save_path, "w") as f:
            for key, value in eval_result.items():
                f.write(key + ":" + str(value) + "\n")

    @staticmethod
    def check_inputs_for_nlgeval(
        hypothesis: PathLike, references: Union[str, List[str]]
    ):
        file_extension = pathlib.Path(references).suffix
        assert file_extension == ".txt", "only support .txt extension"
        hypothesis_text = READ_FILE_FN[file_extension](hypothesis)
        references_text = READ_FILE_FN[file_extension](references)
        assert len(hypothesis_text) == len(
            references_text
        ), "the hypothesis data length must be the references data length"

        references = [references] if type(references) is str else references
        return references


def get_predictions(
    task,
    model,
    tokenizer,
    data_loader,
    num_beams=4,
    max_length=32,
    length_penalty=1,
    device=torch.device("cpu"),
):
    predictions = []
    questions = []
    with torch.no_grad():
        for batch in tqdm(data_loader):
            outputs = model.generate(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                num_beams=num_beams,
                max_length=max_length,
                length_penalty=length_penalty,
            )

            question = [
                tokenizer.decode(ids, skip_special_tokens=True)
                for ids in batch["labels"]
            ]
            prediction = [
                tokenizer.decode(ids, skip_special_tokens=True) for ids in outputs
            ]
            if task == "mc":
                prediction = [item.split("<sep>") for item in prediction]
            questions.extend(question)
            predictions.extend(prediction)

    return questions, predictions


def generate(args: EvalArguments):
    device = torch.device("cpu")
    if args.device == "cuda" or args.device == "gpu":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = T5Tokenizer.from_pretrained(
        args.tokenizer_name_or_path
        if args.tokenizer_name_or_path
        else args.model_name_or_path
    )
    start_time = time.time()
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name_or_path)
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)

    model.eval()
    logger.info(
        "=" * 10
        + "Load model runtime: "
        + str(round((time.time() - start_time) * 1000, 2))
        + "=" * 10
    )

    start_time = time.time()
    valid_dataset = torch.load(args.valid_file_path)
    collator = Text2TextDataCollator(
        tokenizer=tokenizer, model_type=args.model_type, mode="inference"
    )
    loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=args.eval_batch_size, collate_fn=collator
    )

    questions, predictions = get_predictions(
        task=args.task,
        model=model,
        tokenizer=tokenizer,
        data_loader=loader,
        num_beams=args.num_beams,
        max_length=args.max_decoding_length,
        device=device,
    )
    if args.task == "mc":
        predictions = [
            " <sep> ".join(predictions[idc]) for idc in range(len(predictions))
        ]

    logger.info(
        "=" * 10
        + "Inference runtime: "
        + str(round((time.time() - start_time) * 1000, 2))
        + "=" * 10
    )

    # Save ground truth reference text strings
    if args.reference_path:
        with open(args.reference_path, "w") as f:
            f.write("\n".join(questions))
        logger.info(f"Reference saved at {args.reference_path}")

    with open(args.output_path, "w") as f:
        f.write("\n".join(predictions))
        logger.info(f"Output saved at {args.output_path}")


def main():
    parser = HfArgumentParser((EvalArguments,))
    args = parser.parse_args_into_dataclasses()[0]
    generate(args=args)


if __name__ == "__main__":
    main()
