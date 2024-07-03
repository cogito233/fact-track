# Directly Copy Pasted from chatGPT_API.py
# Need to modify to simplize the code and remove the dependency on story_generation
# Advanced todo: add more choices of llama model and others (e.g. OPT-175B)

import time
import logging
import json

import torch
from transformers import AutoTokenizer
from openai import OpenAI
import openai

import os
client = OpenAI(api_key=os.environ['OPENAI_API_KEY'],
                base_url="https://api.openai.com/v1")

def cut_last_sentence(text): # remove possibly incomplete last sentence
    text = text.rstrip() + ' and' # possibly start a new sentence so we can delete it, if the last sentence is already complete and ended with a period
    last_sentence = split_paragraphs(text, mode='sentence')[-1].strip() # possibly incomplete, so strip it
    text = text.rstrip()[:len(text.rstrip()) - len(last_sentence)].rstrip()
    return text


GPT3_END = 'THE END.'
PRETRAINED_MODELS = ['ada', 'babbage', 'curie', 'davinci', 'text-ada-001', 'text-babbage-001', 'text-curie-001',
                     'text-davinci-001', 'text-davinci-002', 'text-davinci-003']
# import os
# if 'OPENAI_API_KEY' in os.environ:
#     openai.api_key = os.environ['OPENAI_API_KEY']
# else:
#     path = "/cluster/home/zzhiheng/cogito/openai_key/openai_key_berkeley.txt"
#     openai.api_key = open(path, 'r').read().strip()
#     # print(openai.api_key)

class ChatGPT3Summarizer(object):
    def __init__(self, args):
        assert args.gpt3_model is not None
        self.model = args.gpt3_model
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.args = args
        self.controller = None
        self.summarize = {
            "num_queries": 0,
            "total_prompt_tokens": 0,
            "total_output_tokens": 0,
            "maximal_prompt_tokens": 0,
        }

    @torch.no_grad()
    def __call__(self, texts, suffixes=None, max_tokens=None, top_p=None, temperature=None, retry_until_success=True,
                 stop=None, logit_bias=None, num_completions=1, cut_sentence=False, model_string=None):
        assert type(texts) == list
        self.summarize['num_queries'] += len(texts)
        if logit_bias is None:
            logit_bias = {}
        if suffixes is not None:
            raise NotImplementedError
        if model_string is None:
            pass
            #logging.warning('model string not provided, using default model')
        else:
            #logging.warning('model string provided, but not used for chat gpt3 summarizer')
            model_string = None
        if self.controller is None:
            return self._call_helper(texts, max_tokens=max_tokens, top_p=top_p,
                                     temperature=temperature, retry_until_success=retry_until_success, stop=stop,
                                     logit_bias=logit_bias, num_completions=num_completions, cut_sentence=cut_sentence,
                                     model_string=model_string)
        else:
            raise NotImplementedError

    @torch.no_grad()
    def _call_helper(self, texts, max_tokens=None, top_p=None, temperature=None,
                     retry_until_success=True, stop=None, logit_bias=None, num_completions=1, cut_sentence=False,
                     model_string=None):
        assert model_string in PRETRAINED_MODELS or model_string is None

        if logit_bias is None:
            logit_bias = {}

        outputs = []
        for i in range(len(texts)):
            text = texts[i]
            prompt = text

            retry = True
            num_fails = 0
            while retry:
                try:
                    context_length = len(self.tokenizer.encode(prompt))
                    self.summarize['total_prompt_tokens'] += context_length
                    self.summarize['maximal_prompt_tokens'] = max(self.summarize['maximal_prompt_tokens'], context_length)
                    if context_length > self.args.max_context_length:
                        logging.warning('context length' + ' ' + str(
                            context_length) + ' ' + 'exceeded artificial context length limit' + ' ' + str(
                            self.args.max_context_length))
                        time.sleep(5)  # similar interface to gpt3 query failing and retrying
                        assert False
                    if max_tokens is None:
                        max_tokens = min(self.args.max_tokens, self.args.max_context_length - context_length)
                    engine = self.model if model_string is None else model_string
                    if engine == 'text-davinci-001':
                        engine = 'text-davinci-002'  # update to latest version
                    logging.log(21, 'PROMPT')
                    logging.log(21, prompt)
                    logging.log(21, 'MODEL STRING:' + ' ' + self.model if model_string is None else model_string)
                    #print(temperature if temperature is not None else self.args.summarizer_temperature)
                    completion = client.chat.completions.create(model=engine,
                    messages=[
                        {'role': 'user', 'content': prompt}
                    ],
                    #suffix=suffixes[i] if suffixes is not None else None,
                    max_tokens=max_tokens,
                    temperature=temperature if temperature is not None else self.args.summarizer_temperature,
                    #top_p=top_p if top_p is not None else self.args.summarizer_top_p,
                    #frequency_penalty=self.args.summarizer_frequency_penalty,
                    #presence_penalty=self.args.summarizer_presence_penalty,
                    stop=stop,
                    logit_bias=logit_bias,
                    n=num_completions)
                    #print(completion)
                    #exit(0)
                    gpt3_pair = {'prompt': prompt, 'completion': [completion.choices[j].message.content for j in range(num_completions)]}
                    logfile = open('gpt3_log.txt', 'a')
                    logfile.write(json.dumps(gpt3_pair) + '\n')
                    logfile.close()
                    retry = False
                except Exception as e:
                    logging.warning(str(e))
                    retry = retry_until_success
                    num_fails += 1
                    if num_fails > 20:
                        raise e
                    if retry:
                        logging.warning('retrying...')
                        time.sleep(num_fails)
            outputs += [completion.choices[j].message.content for j in range(num_completions)]
        if cut_sentence:
            for i in range(len(outputs)):
                if len(outputs[i].strip()) > 0:
                    outputs[i] = cut_last_sentence(outputs[i])
        engine = self.model if model_string is None else model_string
        logging.log(21, 'OUTPUTS')
        logging.log(21, str(outputs))
        logging.log(21, 'GPT3 CALL' + ' ' + engine + ' ' + str(
            len(self.tokenizer.encode(texts[0])) + sum([len(self.tokenizer.encode(o)) for o in outputs])))
        self.summarize['total_output_tokens'] += sum([len(self.tokenizer.encode(o)) for o in outputs])
        return outputs

def load_model(temp = 0.5):
    import argparse
    # set openai.api_key to environment variable OPENAI_API_KEY
    import os
    args = argparse.Namespace()
    args.gpt3_model = 'gpt-3.5-turbo'
    args.max_tokens = 1024 # output length
    args.max_context_length = 2048 # input length
    args.summarizer_temperature = temp
    #args.summarizer_temperature = 0.5
    args.summarizer_frequency_penalty = 0.0
    args.summarizer_presence_penalty = 0.0
    gpt3 = ChatGPT3Summarizer(args)
    return gpt3

def load_model2classification(model = 'gpt-3.5-turbo'):
    import argparse
    # set openai.api_key to environment variable OPENAI_API_KEY
    import os
    args = argparse.Namespace()
    if model == 'gpt-4':
        args.gpt3_model = 'gpt-4-1106-preview'
    elif model == 'gpt-4-turbo':
        args.gpt3_model = 'gpt-4-turbo'
    elif model == 'gpt-3.5-turbo':
        args.gpt3_model = 'gpt-3.5-turbo'
    else:
        raise Exception("model not supported")
    args.max_tokens = 1024 # output length
    args.max_context_length = 2048 # input length
    args.summarizer_temperature = 0
    gpt3 = ChatGPT3Summarizer(args)
    return gpt3

def load_model2generation(temp = 0, model_name = 'gpt-4'):
    import argparse
    # set openai.api_key to environment variable OPENAI_API_KEY
    import os
    args = argparse.Namespace()
    if model_name == "gpt-4-turbo":
        args.gpt3_model = 'gpt-4-turbo'
    elif model_name == 'gpt-4':
        args.gpt3_model = 'gpt-4-1106-preview'
    elif model_name == 'gpt-3.5-turbo':
        args.gpt3_model = 'gpt-3.5-turbo'
    else:
        raise Exception("model not supported")
    #args.gpt3_model = 'gpt-3.5-turbo'
    args.max_tokens = 4096 # output length
    args.max_context_length = 4096 # input length
    args.summarizer_temperature = temp
    gpt3 = ChatGPT3Summarizer(args)
    return gpt3

def determistic_simple_API(model, text, logit_bias = None):
    ChatList = [{'role': 'user', 'content': text}]
    if logit_bias == None:
        logit_bias = {}
    response = client.chat.completions.create(model=model,
    messages=ChatList,
    temperature = 0,
    logit_bias = logit_bias).choices[0].message.content
    return response

def non_determistic_simple_API(model, text, logit_bias = None, temp = 0.1):
    # from openai import OpenAI
    #
    # api_key = "sk-PTNIyjsGWiSBopXMB43896E4E6F341F4B5133d7c1fC0D6E4"
    # base_url = "https://api.132006.xyz/v1"
    #
    # client = OpenAI(
    #     api_key=api_key,
    #     base_url=base_url
    # )

    ChatList = [{'role': 'user', 'content': text}]
    if logit_bias == None:
        logit_bias = {}
    # response = openai.ChatCompletion.create(
    response = client.chat.completions.create(
        model=model,
        messages=ChatList,
        temperature = temp,
        logit_bias = logit_bias,
    )
    # print(response)
    response = response.choices[0].message.content
    return response

def openai_UnitContradictCheck(fact1, fact2, model = None):
    if model == None:
        raise Exception("model is not loaded")
    prompt = f"Do the following statements contradict each other? Answer \"Yes\" or \"No\".\n\n{fact1}\n{fact2}\n"
    answer = model([prompt])[0]
    # print(prompt, answer)
    return True if "Yes" in answer else False

if __name__=='__main__':
    import argparse
    # set openai.api_key to environment variable OPENAI_API_KEY
    import os
    #openai.api_key = os.environ['OPENAI_API_KEY']
    #gpt3 = load_model()
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
    #outputs = gpt3(texts, num_completions=1)
    #print(outputs[0])
    #print(gpt3.summarize)
    print(determistic_simple_API('gpt-4-1106-preview', texts[0]))

















"""
curl http://localhost:9001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Llama-2-7b-chat-hf",
    "messages": [
      {
        "role": "system",
        "content": "You are a helpful assistant."
      },
      {
        "role": "user",
        "content": "Hello!"
      }
    ]
  }'
"""