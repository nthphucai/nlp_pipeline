# coding=utf-8
import torch
from typing import Optional, Union
from dataclasses import dataclass, field

from peft import AutoPeftModelForCausalLM, PeftModel
from transformers import (
    BloomForCausalLM,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    BartTokenizer,
    HfArgumentParser,
    AutoTokenizer,
    T5Tokenizer,
    LlamaTokenizer,
    BloomTokenizerFast,
)

from typing import Optional
from threading import Thread

from nlp_pipeline.utils.file_utils import read_yaml_file
from nlp_pipeline.modules.gen_llm.generate import Generator
from nlp_pipeline.models.qlora import PerfModelConfig

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
    "llama": AutoPeftModelForCausalLM,
}


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        default="meta-llama/Llama-2-7b-hf",
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        },
    )

    tokenizer_name_or_path: Optional[str] = field(
        default="output/customed_tokenizer/llama",
        metadata={
            "help": "Pretrained tokenizer name or path if not the same as model_name"
        },
    )
    cache_dir: Optional[str] = field(
        default="output/models/adapter_llama2",
        metadata={
            "help": "Where do you want to store the pretrained models downloaded from s3"
        },
    )

    adapter_dir: Optional[str] = field(
        default="output/models/adapter_llama2",
        metadata={
            "help": "Path to adapter weight"
        },
    )

    model_type: str = field(
        default="llama",
        metadata={"help": "One of 't5', 'llama', 'bart'"},
    )

    config_dir: Optional[str] = field(
        default="configs/generate.yaml",
        metadata={"help": "Path to config file"},
    )


def generate_response_stream(query: str, llm_generator):
    prompt = llm_generator.query_to_prompt(query=query)

    generation_kwargs = dict(prompt=prompt)
    thread = Thread(target=llm_generator, kwargs=generation_kwargs)
    thread.start()

    generate_text = ""
    for text in llm_generator.streamer:
        text = llm_generator.postprocess(text)
        generate_text += text
        yield text


def pipeline(
    query: str,
    model: Optional[Union[str]] = None,
    tokenizer: Optional[Union[str]] = None,
    config_path: str = None,
    model_type: str = None,
):
    # Read the configuration for question generation pipeline
    gen_config = read_yaml_file(config_path)["gen_llm"]
    # print(f"\nGeneration Pipeline config:\n{gen_config}")

    llm_generator = Generator(model=model, tokenizer=tokenizer, **gen_config)
    prompt = Generator.query_to_prompt(query=query, model_type=model_type)
    output = llm_generator(prompt=prompt)
    return output


if __name__ == "__main__":
    parser = HfArgumentParser((ModelArguments))
    model_args = parser.parse_args_into_dataclasses()[0]

    tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name_or_path)

    perf_model = PerfModelConfig(lora_configs=None)
    bnb_config = perf_model.create_bnb_config()

    base_model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        quantization_config=bnb_config,
        device_map="auto"
    )

    # ***********************************************************************************************************
    #query = "Pitt: Hey Teddy! Have you received my message?\r\nTeddy: No. An email?\r\nPitt: No. On the FB messenger.\r\nTeddy: Let me check.\r\nTeddy: Yeah. Ta!"
    query = "For both Apple and Android devices, the RCA shall be available in respective app stores"
    query = "The RCA shall maintain a status which shall be 'RCA Connected' when DA3 is connected and 'RCA disconnected' in all other situations"
    
    result = pipeline(
        query=query,
        model=base_model,
        tokenizer=tokenizer,
        config_path="configs/generate.yaml",
        model_type=model_args.model_type,
    )

    print("\n => Base Model:\n", result)

    base_model.resize_token_embeddings(len(tokenizer))
    
    peft_model = PeftModel.from_pretrained(base_model, model_args.adapter_dir, torch_dtype=torch.float16)

    result = pipeline(
        query=query,
        model=peft_model,
        tokenizer=tokenizer,
        config_path="configs/generate.yaml",
        model_type=model_args.model_type,
    )
    print("\n => Peft Model:\n", result)
