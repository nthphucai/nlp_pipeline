import os

import torch
import bitsandbytes as bnb
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    AutoPeftModelForCausalLM,
)
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


class PerfModelConfig:
    """
    Reference: https://blog.ovhcloud.com/fine-tuning-llama-2-models-using-a-single-gpu-qlora-and-ai-notebooks/
    """

    def __init__(self, lora_configs):
        self.lora_rank = 16
        self.lora_alpha = 64
        self.lora_dropout = 0.1
        self.bias = "none"

        self.n_gpus = torch.cuda.device_count()
        self.max_memory = f"{40960}MB"

    def __call__(self, model, peft_config):
        # 1 - Enabling gradient checkpointing to reduce memory usage during fine-tuning
        # model.gradient_checkpointing_enable()
        # 2 - Using the prepare_model_for_kbit_training method from PEFT
        model = prepare_model_for_kbit_training(model)

        # Create PEFT config for these modules and wrap the model to PEFT
        model = get_peft_model(model, peft_config)

        # Print information about the percentage of trainable parameters
        self.print_trainable_parameters(model)
        return model

    @staticmethod
    def create_bnb_config():
        """
        Create a bitsandbytes configuration
        Load our LLM in 4 bits, apply bfloat16 compute data type
        and nested quantization for memory-saving purposes
        """
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        return bnb_config

    def create_peft_config(self, modules=["q_proj", "k_proj", "v_proj", "o_proj"]):
        """
        Create Parameter-Efficient Fine-Tuning config for your model
        :param modules: Names of the modules to apply Lora to
        """
        lora_config = LoraConfig(
            r=self.lora_rank,  # dimension of the updated matrices
            lora_alpha=self.lora_alpha,  # parameter for scaling
            target_modules=modules,
            lora_dropout=self.lora_dropout,  # dropout probability for layers
            bias=self.bias,
            task_type="CAUSAL_LM",
        )
        return lora_config


    @staticmethod
    def merge_and_save_model(output_cp_dir: str, output_merged_dir: str):
        #### Merge perf and base model ####
        model = AutoPeftModelForCausalLM.from_pretrained(
            output_cp_dir, device_map="auto", torch_dtype=torch.bfloat16
        )
        model = model.merge_and_unload()
        model.save_pretrained(output_merged_dir, safe_serialization=True)
        print("Saving final merged checkpoint:", output_merged_dir)

    @staticmethod
    def find_all_linear_names(model):
        """
        Previous function needs the target modules to update the necessary matrices.
        The following function will get them for our model:

        """
        cls = (
            bnb.nn.Linear4bit
        )  # if args.bits == 4 else (bnb.nn.Linear8bitLt if args.bits == 8 else torch.nn.Linear)
        lora_module_names = set()
        for name, module in model.named_modules():
            if isinstance(module, cls):
                names = name.split(".")
                lora_module_names.add(names[0] if len(names) == 1 else names[-1])

        if "lm_head" in lora_module_names:  # needed for 16-bit
            lora_module_names.remove("lm_head")
        return list(lora_module_names)

    @staticmethod
    def print_trainable_parameters(model, use_4bit=False):
        """
        Prints the number of trainable parameters in the model.
        """
        trainable_params = 0
        all_param = 0
        for _, param in model.named_parameters():
            num_params = param.numel()
            # if using DS Zero 3 and the weights are initialized empty
            if num_params == 0 and hasattr(param, "ds_numel"):
                num_params = param.ds_numel

            all_param += num_params
            if param.requires_grad:
                trainable_params += num_params
        if use_4bit:
            trainable_params /= 2
        print(
            f"all params: {all_param:,d} || trainable params: {trainable_params:,d} || trainable%: {100 * trainable_params / all_param}"
        )


def run_test():
    bnb_config = PerfModel.create_bnb_config()

    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-2-7b-hf",
        cache_dir="output/models",
        device_map="auto",  # "balanced",
        load_in_8bit=True,
        bnb_config=bnb_config,
        torch_dtype=torch.float16,
    )
    lora_configs = ""
    model = PerfModel(lora_configs)(model=model)
    return model
