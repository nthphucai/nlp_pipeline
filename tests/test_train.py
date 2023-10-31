import bitsandbytes as bnb
import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, LlamaForCausalLM, LlamaTokenizer
from transformers import Trainer, TrainingArguments
from datasets import load_dataset
from peft import (
        get_peft_model, 
        prepare_model_for_kbit_training, 
        LoraConfig
    )
from trl import SFTTrainer

from nlp_pipeline.dataset.build_transformer_format_dataset.data_collator import (
    Text2TextDataCollator,
)
from nlp_pipeline.models.qlora import PerfModelConfig

tokenizer = LlamaTokenizer.from_pretrained("output/customed_tokenizer/llama")

perf_model = PerfModelConfig(lora_configs=None)

bnb_config = perf_model.create_bnb_config()
peft_config = perf_model.create_peft_config()

model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf",
                                            # load_in_8bit=True,
                                            quantization_config=bnb_config,
                                             device_map="auto"
                                        )

model.config.use_cache = False
model.resize_token_embeddings(len(tokenizer))
model = perf_model(model=model, peft_config=peft_config)


output_dir = "/content/drive/MyDrive/NLP/nlp_pipeline/output/models"
per_device_train_batch_size = 4
gradient_accumulation_steps = 4
per_device_eval_batch_size = 4
eval_accumulation_steps = 4
optim = "paged_adamw_32bit"
save_steps = 10
logging_steps = 10
learning_rate = 5e-4
max_grad_norm = 0.3
max_steps = 50
warmup_ratio = 0.03
evaluation_strategy="steps"
lr_scheduler_type = "constant"

training_args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            optim=optim,
            evaluation_strategy=evaluation_strategy,
            save_steps=save_steps,
            learning_rate=learning_rate,
            logging_steps=logging_steps,
            max_grad_norm=max_grad_norm,
            max_steps=max_steps,
            warmup_ratio=warmup_ratio,
            # group_by_length=True,
            lr_scheduler_type=lr_scheduler_type,
            # ddp_find_unused_parameters=False,
            eval_accumulation_steps=eval_accumulation_steps,
            per_device_eval_batch_size=per_device_eval_batch_size,
            remove_unused_columns=False
        )


train_dataset = torch.load("/content/drive/MyDrive/NLP/nlp_pipeline/dataset/train_data_hl_t5.pt")

valid_dataset = torch.load("/content/drive/MyDrive/NLP/nlp_pipeline/dataset/valid_data_hl_t5.pt")

data_collator = Text2TextDataCollator(
      tokenizer=tokenizer,
      llm_architect="decoder-only",
      model_type="llama",
      mode="training",
      using_tpu=False
  )

trainer = Trainer(
    model=model, 
    args=training_args, 
    train_dataset=train_dataset, 
    eval_dataset=valid_dataset, 
    data_collator=data_collator
  )

trainer.train()
trainer.save_model(f"{output_dir}/final")

