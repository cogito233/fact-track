import os
import json
import requests
from transformers import AutoTokenizer
import torch

TOGETHER_API_KEY = os.environ.get("TOGETHER_API_KEY")

payload = {
    "model": "togethercomputer/Llama-2-7B-32K-Instruct",
    "prompt": "[INST]\nWrite a poem about cats\n[/INST]\n\n",
    "max_tokens": 1000,
    "stop": ["</s>", "[/INST]", "\n\n\n\n"],
    "temperature": 0,
    "top_p": 0.7,
    "top_k": 50,
    "repetition_penalty": 1,
    "n": 1
}
headers = {
    "accept": "application/json",
    "content-type": "application/json",
    "Authorization": f"Bearer {TOGETHER_API_KEY}"
}

# response = requests.post(url, json=payload, headers=headers)
#
# # print(response)
# print(parse_string(response.text))


class Llama32KSummarizer(object):
    def __init__(self, args):
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.args = args
        self.api_key = TOGETHER_API_KEY  # API key for model access
        self.url = "https://api.together.xyz/completions"  # URL for the model API
        self.summarize = {
            "num_queries": 0,
            "total_prompt_tokens": 0,
            "total_output_tokens": 0,
            "maximal_prompt_tokens": 0,
        }

    @torch.no_grad()
    def __call__(self, texts, max_tokens=1024, top_p=0.7, temperature=0, retry_until_success=True, stop=["</s>", "[/INST]", "[INST]", "\n\n\n\n"],
                 logit_bias=None, num_completions=1):
        assert type(texts) == list

        responses = []
        for text in texts:
            payload = self._prepare_payload(text, max_tokens, top_p, temperature, stop, logit_bias, num_completions)
            response = self._send_request(payload)
            parsed_response = self._parse_response(response)
            prompt_tokens = len(self.tokenizer.tokenize(text))
            output_tokens = len(self.tokenizer.tokenize(parsed_response))
            self.summarize['num_queries'] += 1
            self.summarize['total_prompt_tokens'] += prompt_tokens
            self.summarize['total_output_tokens'] += output_tokens
            self.summarize['maximal_prompt_tokens'] = max(self.summarize['maximal_prompt_tokens'], prompt_tokens)
            responses.append(parsed_response)

        return responses

    def _prepare_payload(self, text, max_tokens, top_p, temperature, stop, logit_bias, num_completions):
        payload = {
            "model": "togethercomputer/Llama-2-7B-32K-Instruct",
            "prompt": "[INST]\n" + text + "\n[/INST]\n\n",
            "max_tokens": max_tokens or self.args.max_tokens,
            "stop": stop,
            "temperature": temperature or self.args.summarizer_temperature,
            "top_p": top_p,
            "repetition_penalty": 1,
            "n": num_completions
        }
        # print(payload)
        return payload

    def _send_request(self, payload):
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        response = requests.post(self.url, json=payload, headers=headers)
        return response.json()

    def _parse_response(self, response):
        # Implement logic to parse and format the response from the model
        # input_string = response.text
        # # 将字符串转换为JSON对象
        # data = json.loads(input_string)
        # 提取并返回所需的文本部分
        # print(response)
        return response['choices'][0]['text']

def load_model_llama(temp = 0.5):
    import argparse
    # set openai.api_key to environment variable OPENAI_API_KEY
    import os
    args = argparse.Namespace()
    args.max_tokens = 10240 # output length
    args.max_context_length = 20480 # input length
    args.summarizer_temperature = temp
    #args.summarizer_temperature = 0.5
    args.summarizer_frequency_penalty = 0.0
    args.summarizer_presence_penalty = 0.0
    llama = Llama32KSummarizer(args)
    return llama

if __name__=='__main__':
    import argparse
    import os
    llama = load_model_llama()
    texts = ["""Premise: An ordinary high school student discovers that they possess an extraordinary ability to manipulate reality through their dreams.    As they struggle to control this power and keep it hidden from those who would exploit it, they are drawn into a dangerous conflict between secret organizations vying for control over the fate of the world.

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
"""]
    outputs = llama(texts, num_completions=1)
    print(outputs[0])
    print(llama.summarize)


