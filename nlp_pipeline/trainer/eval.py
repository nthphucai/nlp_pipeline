import pathlib
import time
from dataclasses import dataclass, field
from os import PathLike
from typing import Dict, List, Optional, Union

import torch
from tqdm.auto import tqdm
from transformers import (
    AutoConfig,
    LlamaForCausalLM,
    BloomForCausalLM,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    BartTokenizer,
    HfArgumentParser,
    T5Tokenizer,
    LlamaTokenizer,
    BloomTokenizerFast,
    GenerationConfig,
    set_seed,
)
from peft import PeftModel
from transformers.generation.utils import GreedySearchDecoderOnlyOutput

from nlgeval import compute_metrics as eval_automatic

from test_automation.dataset.build_transformer_format_dataset.data_collator import (
    Text2TextDataCollator,
)
from test_automation.utils import READ_FILE_FN
from test_automation.utils.file_utils import logger

MODEL_TYPE_TO_TOKENIZER = {
    "t5": T5Tokenizer,
    "bart": BartTokenizer,
    "mixtral": LlamaTokenizer,
    "bloom": BloomTokenizerFast,
    "llama": LlamaTokenizer,
}

MODEL_TYPE_TO_LLM = {
    "t5": AutoModelForSeq2SeqLM,
    "bart": AutoModelForSeq2SeqLM,
    "mixtral": AutoModelForCausalLM,
    "bloom": BloomForCausalLM,
    "llama": LlamaForCausalLM,
}


@dataclass
class EvalArguments:
    model_name_or_path: str = field(
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        }
    )
    adapter_model: str = field(
        metadata={
            "help": "Path to adapter model or model identifier from huggingface.co/models"
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


# def get_predictions(
#     model,
#     tokenizer,
#     data_loader,
#     num_beams=4,
#     max_length=32,
#     length_penalty=1,
#     device=torch.device("cpu"),
# ):
#     predictions = []
#     requirements = []
#     with torch.no_grad():
#         for batch in tqdm(data_loader):
#             outputs = model.generate(
#                 input_ids=batch["input_ids"].to(device),
#                 attention_mask=batch["attention_mask"].to(device),
#                 num_beams=num_beams,
#                 max_length=max_length,
#                 length_penalty=length_penalty,
#             )

#             requirement = [
#                 tokenizer.decode(ids, skip_special_tokens=True)
#                 for ids in batch["labels"]
#             ]
#             prediction = [
#                 tokenizer.decode(ids, skip_special_tokens=True) for ids in outputs
#             ]
#             requirements.extend(requirement)
#             predictions.extend(prediction)

#     return requirements, predictions


def get_predictions(
    model,
    tokenizer,
    data_loader,
    num_beams=4,
    max_length=32,
    length_penalty=1.1,
    device=torch.device("cpu"),
):
    encoding = tokenizer(data_loader, return_tensors="pt")
    input_ids = encoding["input_ids"].to(device)

    generation_config = GenerationConfig(
        temperature=0.1,
        top_p=0.75,
        num_beams=num_beams,
        max_length=max_length,
        repetition_penalty=length_penalty,
    )
    with torch.inference_mode():
        for batch in tqdm(data_loader):
            return model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=256,
            )


def generate(args: EvalArguments):
    device = torch.device("cpu")
    if args.device == "cuda" or args.device == "gpu":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = MODEL_TYPE_TO_TOKENIZER[args.model_type].from_pretrained(
        args.tokenizer_name_or_path
        if args.tokenizer_name_or_path
        else args.model_name_or_path
    )

    start_time = time.time()
    model = MODEL_TYPE_TO_LLM[args.model_type].from_pretrained(
        args.model_name_or_pathc, load_in_8bit=True, device_map="cuda"
    )

    # model.resize_token_embeddings(len(tokenizer))
    model = PeftModel.from_pretrained(
        model, args.adapter_model, torch_dtype=torch.float16
    )

    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2

    model = model.eval()
    model = torch.compile(model)

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
        model=model,
        tokenizer=tokenizer,
        data_loader=loader,
        num_beams=args.num_beams,
        max_length=args.max_decoding_length,
        device=device,
    )

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
