import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

def create_bnb_config():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    return bnb_config

def load_model(model_name, bnb_config):
    n_gpus = torch.cuda.device_count()
    max_memory = f'{40960}MB'

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        max_memory = {i: max_memory for i in range(n_gpus)},
    )
    return model

bnb_config = create_bnb_config()
model = load_model(model_name="meta-llama/Llama-2-7b-hf", bnb_config=bnb_config)

from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

# Prompt text
prompt = "For both Apple and Android devices, the RCA shall be available in respective app stores"

# Encode the prompt text
input_ids = tokenizer.encode(prompt, return_tensors="pt")
input_ids = input_ids.to("cuda")

# Generate response with maximum length of 50 tokens
output = model.generate(input_ids, max_length=50, do_sample=True)

# Decode the generated tokens
response = tokenizer.decode(output[0], skip_special_tokens=True)

print(response)
