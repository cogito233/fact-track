# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import fire

import torch
from vllm import LLM
from vllm import LLM, SamplingParams

torch.cuda.manual_seed(42)
torch.manual_seed(42)

model_name = "7B-chat"
max_gen_len = 40960
# temperature = 0.5
temperature = 0
top_p = 1
from llama_chat_api_vLLM import LlamaPipeline
model = LlamaPipeline(model_name = model_name, max_gen_len = max_gen_len, temperature = temperature, top_p = top_p)

class LLama2Generator(object):
    def __init__(self, model_name = "7B-chat", max_gen_len = 40960, temperature = 0, top_p = 1):
        self.model = model
        # Since it is hard to handle with CUDA memory leak, so we use the same model for all the requests
        self.summarize = {
            "num_queries": 0,
            "total_prompt_tokens": 0,
            "total_output_tokens": 0,
            "maximal_prompt_tokens": 0,
        }

    @torch.no_grad()
    def __call__(self, prompts): # input is a list of texts
        dialogues = []
        for prompt in prompts:
            dialogues.append([
                {
                "role": "user",
                "content": prompt
                }
            ])
        outputs = self.model.inference(dialogues)
        outputs = [output["content"] for output in outputs]
        return outputs

    def delete(self):
        pass
        

def load_deterministic_llama2(model_name = "7B-chat"):
    model = LLama2Generator(model_name = model_name)
    return model

def load_creative_model(model_name = "7B-chat", temp=0.5):
    # Temp = 0.5
    model = LLama2Generator(model_name = model_name, temperature = 0.5)
    return model

if __name__ == "__main__":
    model = load_deterministic_llama2()
    texts = """Premise: An ordinary high school student discovers that they possess an extraordinary ability to manipulate reality through their dreams.    As they struggle to control this power and keep it hidden from those who would exploit it, they are drawn into a dangerous conflict between secret organizations vying for control over the fate of the world.

Outline:

Point 2.1.2
Main plot: Alex struggles to control their power
Begin Event: Alex accidentally manipulates reality in their dream
End Event: Alex seeks guidance from Mr. Lee to control their power
Characters: Alex, Mr. Lee


Can you break down point 2.1.2 into less than 3 independent, chronological and same-scaled outline points? Also, assign each character a name. Please use the following template with "Main Plot", "Begin Event". "End Event" and "Characters". Do not answer anything else.

Point 2.1.2.1
Main plot: [TODO]
Begin Event: [TODO]
End Event: [TODO]
Characters: [TODO]

Point 2.1.2.2
Main plot: [TODO]
Begin Event: [TODO]
End Event: [TODO]
Characters: [TODO]

...
"""
    outputs = model([texts, texts])
    print(outputs[0])
    print("#"*50)
    print(outputs[1])