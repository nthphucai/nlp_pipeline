import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional, Union

import torch
import wandb
from trl import SFTTrainer
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
    TrainingArguments,
    Trainer,
    set_seed,
)

from nlp_pipeline.dataset.build_transformer_format_dataset.data_collator import Text2TextDataCollator
from nlp_pipeline.models.qlora import PerfModelConfig
from nlp_pipeline.modules.gen_llm.generate import Generator
from nlp_pipeline.utils.file_utils import logger, read_yaml_file
from nlp_pipeline.utils.model_utils import assert_not_all_frozen, freeze_embeds
from nlp_pipeline.utils.utils import set_gpu


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


TRAINING_ARGS_MAP = {
    "onnx": "ORTSeq2SeqTrainingArguments",
    "default": TrainingArguments,
}

TRAINER_TYPE = {"sftt": SFTTrainer, "base": Trainer}


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        default="t5-small",
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        },
    )
    model_type: str = field(
        default="llama",
        metadata={"help": "One of 't5', 'llama', 'bart'"},
    )
    llm_architect: str = field(
        default="decoder-only",
        metadata={
            "help": "Which llm architecture, one of 'decoder-only', 'encoder-only', 'encoder-decoder'"
        },
    )
    tokenizer_name_or_path: Optional[str] = field(
        default="output/customed_tokenizer",
        metadata={
            "help": "Pretrained tokenizer name or path if not the same as model_name"
        },
    )
    cache_dir: Optional[str] = field(
        default="output/models",
        metadata={
            "help": "Where do you want to store the pretrained models downloaded from s3"
        },
    )
    label_smoothing: Optional[float] = field(
        default=0,
        metadata={
            "help": "Label smoothing rate, set to > 0 if you want to enable lable smoothing"
        },
    )
    freeze_embeds: bool = field(
        default=False,
        metadata={
            "help": "Freeze token embeddings and positional embeddings for BART, just token embeddings for T5."
        },
    )

    trainer_type: str = field(
        default="base",
        metadata={"help": "Which trainer type to use, one of `base trainer`, `sftt - supervised fine-tuning trainer`"},
    )

    project_name: str = field(
        default="question-generation",
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    train_file_path: str = field(
        default="output/dataset/train_data_hl_t5.pt",
        metadata={"help": "Path for cached train dataset"},
    )
    valid_file_path: str = field(
        default="output/dataset/valid_data_hl_t5.pt",
        metadata={"help": "Path for cached valid dataset"},
    )
    train_file_path_new: str = field(
        default=None, metadata={"help": "Path for new cached train dataset"}
    )
    valid_file_path_new: str = field(
        default=None, metadata={"help": "Path for new cached valid dataset"}
    )
    max_source_length: Optional[int] = field(
        default=512, metadata={"help": "Max input length for the source text"}
    )
    max_target_length: Optional[int] = field(
        default=128, metadata={"help": "Max input length for the target text"}
    )


def start_train(
    model_args: ModelArguments,
    data_args: DataTrainingArguments,
    training_args: TrainingArguments):
    assert model_args.model_type in list(
        MODEL_TYPE_TO_TOKENIZER.keys()
    ), "model type should be 't5' or 'bart'"

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. "
            f"Use --overwrite_output_dir to overcome."
        )

    # Setup wandb
    if training_args.report_to == "wandb":
        wandb.login()
        wandb.init(
            project=model_args.project_name,
            name=model_args.model_name_or_path,
            group=model_args.model_type,
            tags=["baseline", model_args.model_type],
            job_type="train",
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters")
    print(training_args)

    # Set seed
    set_seed(training_args.seed)

    # Set project name
    os.environ["WANDB_PROJECT"] = model_args.project_name

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    tokenizer_cls = MODEL_TYPE_TO_TOKENIZER[model_args.model_type]
    tokenizer = tokenizer_cls.from_pretrained(
        (
            model_args.tokenizer_name_or_path
            if model_args.tokenizer_name_or_path
            else model_args.model_name_or_path
        ),
        cache_dir=model_args.cache_dir,
    )
    config = AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )
    model_cls = MODEL_TYPE_TO_LLM[model_args.model_type]

    if model_args.llm_architect == "encoder-decoder":
        model = model_cls.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
            config=config,
        )
        model.resize_token_embeddings(len(tokenizer))
        if model_args.freeze_embeds:
            logger.info("Freezing embeddings of the model")
            freeze_embeds(model)
            assert_not_all_frozen(model)

    elif model_args.llm_architect == "decoder-only":
        n_gpus = torch.cuda.device_count()
        max_memory = f"{40960}MB"

        perf_model = PerfModelConfig(lora_configs=None)

        bnb_config = perf_model.create_bnb_config()
        peft_config = perf_model.create_peft_config()

        base_model = model_cls.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
            config=config,
            quantization_config=bnb_config,
            device_map="auto",
            max_memory={i: max_memory for i in range(n_gpus)},
        )

        base_model_cp = base_model

        base_model.config.use_cache = False
        base_model.resize_token_embeddings(len(tokenizer))
        model = perf_model(model=base_model, peft_config=peft_config)
    else:
        raise ValueError(f"Unknown model type {model_args.model_type}")

    # Get datasets
    logger.info("Loading dataset...")
    train_dataset = (
        torch.load(data_args.train_file_path) if training_args.do_train else None
    )
    valid_dataset = (
        torch.load(data_args.valid_file_path) if training_args.do_eval else None
    )
    logger.info("The number train dataset %s", len(train_dataset))
    logger.info("The number valid dataset %s", len(valid_dataset)) 
    
    # Initialize data_collator
    data_collator = Text2TextDataCollator(
        tokenizer=tokenizer,
        llm_architect=model_args.llm_architect,
        model_type=model_args.model_type,
        mode="training",
        using_tpu=training_args.tpu_num_cores is not None,
    )

    # Initialize our Trainer
    if model_args.trainer_type == "sftt" and model_args.llm_architect == "decoder-only":
        trainer = TRAINER_TYPE["sftt"](
            model=model,
            args=training_args,
            peft_config=peft_config,
            train_dataset=train_dataset,
            eval_dataset=valid_dataset,
        )
    else:
      trainer = TRAINER_TYPE["base"](
              model=model,
              args=training_args,
              train_dataset=train_dataset,
              eval_dataset=valid_dataset,
              data_collator=data_collator,
          )

    # Disable wandb console logs
    logging.getLogger("wandb.run_manager").setLevel(logging.WARNING)

    # Training
    if training_args.do_train:
        trainer.train(
            model_path=(
                model_args.model_name_or_path
                if os.path.isdir(model_args.model_name_or_path)
                else None
            )
        )
    if not os.path.exists(training_args.output_dir):
        os.makedirs(training_args.output_dir)

    trainer.save_model()
    tokenizer.save_pretrained(training_args.output_dir)
    print("Saving last checkpoint of the model at", training_args.output_dir)

    #########################################################################
    gen_config = read_yaml_file("configs/generate.yaml")["gen_llm"]
    llm_generator = Generator(model=model, tokenizer=tokenizer, **gen_config)

    query = "For both Apple and Android devices, the RCA shall be available in respective app stores"
    #query = "Pitt: Hey Teddy! Have you received my message?\r\nTeddy: No. An email?\r\nPitt: No. On the FB messenger.\r\nTeddy: Let me check.\r\nTeddy: Yeah. Ta!"

    prompt = llm_generator.query_to_prompt(query=query, model_type="llama")
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    print("\n => Output from Base model:")
    outputs = base_model_cp.generate(**inputs, max_new_tokens=256)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))

    print("\n => Output from Peft model:")
    outputs = model.generate(**inputs, max_new_tokens=256)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))
    #######################################################################

    # Evaluation
    results = {}
    if training_args.do_eval and training_args.local_rank in [-1, 0]:
        logger.info("*** Evaluate ***")

        eval_output = trainer.evaluate()

        output_eval_file = os.path.join(training_args.output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results *****")
            for key in sorted(eval_output.keys()):
                logger.info("  %s = %s", key, str(eval_output[key]))
                writer.write("%s = %s\n" % (key, str(eval_output[key])))

        results.update(eval_output)

    # Free memory
    del model
    del trainer
    torch.cuda.empty_cache()
    return results


def main(args_file=None):
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments)
    )
    if (len(sys.argv) == 2 and sys.argv[1].endswith(".json")) or args_file is not None:
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        args_file_path = (
            os.path.abspath(sys.argv[1]) if args_file is None else args_file
        )
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=args_file_path
        )
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    return start_train(
        model_args=model_args, data_args=data_args, training_args=training_args
    )


if __name__ == "__main__":
    set_gpu(0)
    main()
