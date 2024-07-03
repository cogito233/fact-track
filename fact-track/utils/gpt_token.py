import tiktoken

enc = tiktoken.encoding_for_model("gpt-4")
assert enc.decode(enc.encode("hello world")) == "hello world"

import transformers
from transformers import GPT2Tokenizer, GPT2LMHeadModel

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

def count_gpt3_tokens(text):
    # TODO: use gpt2 tokenizer to count tokens
    return len(tokenizer.encode(text))

def count_gpt4_tokens(text):
    return len(enc.encode(text))

if __name__ == "__main__":
    print(count_gpt3_tokens("hello world!"))
    print(count_gpt4_tokens("hello world!"))