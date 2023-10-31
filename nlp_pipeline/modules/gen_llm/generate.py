import os

import re
from torch import bfloat16
import torch
from typing import List, Any
from dotenv import load_dotenv

load_dotenv()

from transformers import TextIteratorStreamer
from transformers import BitsAndBytesConfig, StoppingCriteria, StoppingCriteriaList

from .prompt_template import formatted_prompt
from constant import INSTRUCTION

device = "cuda" if torch.cuda.is_available() else "cpu"


class StoppingCriteriaSub(StoppingCriteria):
    def __init__(self, stop_token_ids=[], encounters=1):
        super().__init__()

        self.stop_token_ids = [stop.to(device) for stop in stop_token_ids]

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for stop in self.stop_token_ids:
            if torch.any((stop == input_ids[0][-len(stop) :])).item():
                return True

        return False


class Generator:
    def __init__(self, model, tokenizer, **configs):
        super().__init__()

        self.model = model
        self.model.eval()

        self.tokenizer = tokenizer

        stop_words_ids = ["</s>", "### Instruction", "### Input"]

        stop_token_ids = [
            self.tokenizer(stop_word, return_tensors="pt")["input_ids"].squeeze()
            for stop_word in stop_words_ids
        ]
        self.stopping_criteria = StoppingCriteriaList(
            [StoppingCriteriaSub(stop_token_ids=stop_token_ids, encounters=1)]
        )

        self.generated_request_len = configs.get("generated_request_len", 256)
        self.show_progress_bar = configs.get("show_progress_bar", False)
        self.repetition_penalty = configs.get("repetition_penalty", 1.2)
        self.temperature = configs.get("temperature", 0.2)

        self.streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True)

    def __call__(self, prompt: str) -> Any:
        input_ids = self.tokenizer(prompt, return_tensors="pt")["input_ids"]
        input_ids = input_ids.to(device)

        output = self.model.generate(
            input_ids=input_ids,
            top_p=0.95,
            top_k=0,
            max_new_tokens=self.generated_request_len,
            repetition_penalty=self.repetition_penalty,
            do_sample=True,
            streamer=self.streamer,
            temperature=self.temperature,
            stopping_criteria=self.stopping_criteria,
        )
        generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
        generated_text = generated_text[len(prompt) :].strip()
        return generated_text

    @staticmethod
    def postprocess(generated_text: str):
        generated_text = re.sub(r"<<SYS>>|<</SYS>>", "", generated_text)
        generated_text = re.sub(r"\[.+?\]|\[\]", "", generated_text)
        generated_text = re.sub(r"\<.+?\>|\<\>", "", generated_text)
        generated_text = re.sub(r"\n{2,}", "\n", generated_text)
        generated_text = re.sub(r"###", "\n", generated_text)
        generated_text = re.sub(r"Response|Response:", "", generated_text)
        generated_text = re.sub(r"Input:|Input|Output:|Output", "", generated_text)
        generated_text = re.sub(r"previous_message: " "", "", generated_text)
        generated_text = re.sub(r"</s>", "", generated_text)
        generated_text = (
            generated_text[1:] if generated_text.startswith("\\n") else generated_text
        )
        return generated_text

    @staticmethod
    def query_to_prompt(
        query: str, model_type: str = "instruct_llama2"
    ) -> str:    
        if model_type == "llama":
          """
          Below is an instruction that describes a task, \
          paired with an input that provides further context. \
          Write a response that appropriately completes the request.  
          ### Instruction:
          {}
          ### Input:
          {}
          ### Response:
          {}"""
          prompt = formatted_prompt.format(INSTRUCTION, query, "")

        elif model_type == "instruct_llama2":
            """
            ### Instruction:
            <prompt>

            ### Input:
            <additional context>

            ### Response:
            <leave a newline blank for model to respond>
            """
            conversation = []
            for index, message in enumerate(messages):
                if message["role"] == "system" and index == 0:
                    conversation.append(f"### Instruction\n{message['content']}\n\n")
                elif message["role"] == "user":
                    conversation.append(f"### Input\n{message['content']}\n\n")
                else:
                    conversation.append(f"### Response\n{message['content']}\n\n")

            prompt = "".join(conversation)

        elif model_type == "instruct_llama2_type":
            system_prompt = ""
            user_prompt = "\n\n### Input:"
            messages = ""

            query_prompt = (
                "\n\nFollowing the same format above from the examples, reponse to "
            )

            for index, message in enumerate(messages):
                if message["role"] == "system" and index == 0:
                    system_prompt = f"### Instruction:\n{message['content']}"
                elif message["role"] == "user":
                    user_prompt += f"\nExample: {message['content']}"
                elif message["role"] == "assistant":
                    user_prompt += f"\nResponse: {message['content']}\n"
                else:
                    query_prompt += f"{message['content']}\nResponse:"

            prompt = "".join([system_prompt, user_prompt, query_prompt])
        else:
            raise NotImplementedError(
                "only support prompt template for `base_llama2` and `instruct_llama2`"
            )

        return prompt
