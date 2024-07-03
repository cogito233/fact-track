# from chatGPT_contradictDetector_factDecompose import openai_UnitContradictCheck, huggingface_UnitContradictScore
# from chatGPT_API import load_model2classification
# from outline_analyze import get_embedding_contriever, similarity_from_embedding

# import sys
# sys.path.append(f"{BASE_PATH}/fact-track/fact-track/utils")
# from gpt_api import load_model2classification, openai_UnitContradictCheck
# from huggingface_api import get_embedding_contriever, similarity_from_embedding, huggingface_UnitContradictScore
import os
BASE_PATH = os.environ["BASE_PATH"]
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
tokenizer_contriever = None
model_contriever = None
tokenizer_nli = None
model_nli_fineturned = None
# tokenizer_nli = AutoTokenizer.from_pretrained("MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli")
# finetuned_nli_path = f"{BASE_PATH}/fact-track/models/results_18k"
# tokenizer_nli.save_pretrained(finetuned_nli_path)
def load_models():
    global tokenizer_contriever, model_contriever, tokenizer_nli, model_nli_fineturned
    cotriver_path = f"{BASE_PATH}/fact-track/models/contriever"
    if not os.path.exists(cotriver_path):
        tokenizer_contriever = AutoTokenizer.from_pretrained("facebook/contriever")
        model_contriever = AutoModel.from_pretrained("facebook/contriever")
        model_contriever.save_pretrained(cotriver_path)
        tokenizer_contriever.save_pretrained(cotriver_path)
        model_contriever.to(device)
    else:
        model_contriever = AutoModel.from_pretrained(cotriver_path)
        tokenizer_contriever = AutoTokenizer.from_pretrained(cotriver_path)

    # model_name = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
    # tokenizer_nli = AutoTokenizer.from_pretrained(model_name)
    # model_nli = AutoModelForSequenceClassification.from_pretrained(model_name)
    # model_nli.to(device)
    # this is the base model, but the version we used is by finetuning
    finetuned_nli_path = f"{BASE_PATH}/fact-track/models/results_18k"
    if not os.path.exists(finetuned_nli_path):
        raise Exception("finetuned_nli_path not exists")
    else:
        tokenizer_nli = AutoTokenizer.from_pretrained(finetuned_nli_path)
        model_nli_fineturned = AutoModelForSequenceClassification.from_pretrained(finetuned_nli_path)
        model_nli_fineturned.to(device)
    print("load models done")
    # print(model_contriever)
    # print(model_nli_fineturned)
def get_embedding_contriever(sentence):
    def mean_pooling(token_embeddings, mask):
        token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
        sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
        return sentence_embeddings
    inputs = tokenizer_contriever([sentence], padding=True, truncation=True, return_tensors='pt')
    outputs = model_contriever(**inputs)
    embeddings = mean_pooling(outputs[0], inputs['attention_mask'])
    #print(embeddings.shape)
    return embeddings[0].detach().numpy()

from scipy.spatial.distance import cosine
def similarity_from_embedding(embedding_a, embedding_b):
    similarity = 1 - cosine(embedding_a, embedding_b)
    return similarity

def huggingface_UnitContradictCheck(fact1, fact2, detail = False):
    input = tokenizer_nli(fact1, fact2, truncation=True, return_tensors="pt")
    # print(tokenizer_nli)
    # print(input)
    # print(model_nli_fineturned)
    # change input to gpu
    output = model_nli_fineturned(input["input_ids"].to(device))  # device = "cuda:0" or "cpu"
    prediction = torch.softmax(output["logits"][0], -1).tolist()
    label_names = ["entailment", "neutral", "contradiction"]
    prediction = {name: pred for pred, name in zip(prediction, label_names)}
    # print(prediction)
    if detail == True:
        return prediction["contradiction"]>0.5, prediction
    return prediction["contradiction"] > 0.5

def huggingface_UnitContradictScore(fact1, fact2):
    pred, score = huggingface_UnitContradictCheck(fact1, fact2, detail=True)
    return score["contradiction"]

load_models()
# print(tokenizer_contriever)
# print(model_contriever)
# print(tokenizer_nli)
# print(model_nli_fineturned)

if __name__ == "__main__":
    def save_contriver():
        model.save_pretrained(path)
    fact1 = "Amsterdam is the capital of the Netherlands."
    fact2 = "Amsterdam is in the Netherlands."
    print(huggingface_UnitContradictScore(fact1, fact2))
    embedding_1 = get_embedding_contriever(fact1)
    embedding_2 = get_embedding_contriever(fact2)
    print(similarity_from_embedding(embedding_1, embedding_2))