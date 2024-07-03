from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer
import torch
from torch import nn

import os
os.environ["WANDB_DISABLED"] = "true"
#os.environ["CUDA_VISIBLE_DEVICES"] = "7"

# device = torch.device("cuda:5") if torch.cuda.is_available() else torch.device("cpu")
device = torch.device("cuda:0")
# device = torch.device("cpu")
model_name = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
tokenizer_nli = AutoTokenizer.from_pretrained(model_name)
model_nli = AutoModelForSequenceClassification.from_pretrained(model_name)
model_nli.to(device)


class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        # forward pass
        outputs = model(**inputs)
        # print(inputs.keys())
        # print(outputs.keys())
        # print(outputs.get("logits").shape)
        # print(labels.shape)
        # change false to 0, true to 1
        labels = torch.where(labels == 0, torch.tensor(0).to(device), labels)
        # Map logit from 3 dim to 2 dim, 0,1->0, 2->1
        logits_neg = outputs.get("logits")[:, 0:2]
        # calculate the mean of the two logits
        logits_neg = torch.mean(logits_neg, dim=1).unsqueeze(1)
        logits_pos = outputs.get("logits")[:, 2].unsqueeze(1)
        logits = torch.cat((logits_neg, logits_pos), dim=1)
        #print(logits)
        #print(labels)
        # print(logits.shape)
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, 2), labels.view(-1))
        print(loss)
        #exit(0)
        return (loss, outputs) if return_outputs else loss

def tokenize(batch):
    if batch.get('fact1')==None:
        return tokenizer_nli(batch['fact_content1'], batch['fact_content2'], padding='max_length', truncation=True)
    return tokenizer_nli(batch['fact1'], batch['fact2'], padding='max_length', truncation=True)

def train_model():
    from transformers import TrainingArguments, Trainer
    from finetuning_NLI_data import load_dataset, candidate_path
    transet_path = "/home/yangk/zhiheng/develop_codeversion/chatGPT-planner/analyze_code/dataset/data_csv/factpair_subset.csv"
    train_dataset = load_dataset(transet_path)
    test_dataset = load_dataset(candidate_path[3])
    train_dataset = train_dataset.map(tokenize, batched=True, batch_size=len(train_dataset))
    test_dataset = test_dataset.map(tokenize, batched=True, batch_size=len(test_dataset))
    train_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])
    test_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])
    # print(train_dataset[0])

    training_args = TrainingArguments(
        output_dir='./results',           # 输出目录
        num_train_epochs=1,              # 总共训练的epoch数量
        per_device_train_batch_size=4,  # 每个GPU的训练批次大小
        per_device_eval_batch_size=4,   # 每个GPU的评估批次大小
        warmup_steps=100,                # 预热步数
        weight_decay=0.01,               # 权重衰减
        logging_dir='./logs',            # 日志目录
        # no_cuda=True,                   # 不使用GPU
    )
    trainer = CustomTrainer(
        model=model_nli,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset
    )
    trainer.train()
    trainer.save_model("./results_18k")
    return trainer.evaluate()


# device = torch.device("cuda:7")
#model_fineturned = AutoModelForSequenceClassification.from_pretrained("./results_18k")
#model_fineturned = AutoModelForSequenceClassification.from_pretrained("./results")
model_fineturned = AutoModelForSequenceClassification.from_pretrained(model_name)
model_fineturned.to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)
def huggingface_finetunedContradictScore(fact1, fact2):
    input = tokenizer(fact1, fact2, truncation=True, return_tensors="pt")
    output = model_fineturned(input["input_ids"].to(device))  # device = "cuda:0" or "cpu"
    prediction = torch.softmax(output["logits"][0], -1).tolist()
    label_names = ["entailment", "neutral", "contradiction"]
    prediction = {name: pred for pred, name in zip(prediction, label_names)}
    return prediction["contradiction"]

def evaluate_model():
    from finetuning_NLI_data import load_dataset, candidate_path
    #test_dataset = load_dataset(candidate_path[0])
    transet_path = "/home/yangk/zhiheng/develop_codeversion/chatGPT-planner/analyze_code/dataset/data_csv/factpair_subset.csv"
    testset_path = "/home/yangk/zhiheng/develop_codeversion/chatGPT-planner/analyze_code/injection/outline_metadata/test_gpt4_depth2/test_gpt4_depth2_contradictStat_factContradict.csv"
    test_dataset = load_dataset(testset_path)
    # subsample the dataset
    #test_dataset = test_dataset.select(range(1000))
    #test_dataset = test_dataset.map(tokenize, batched=True, batch_size=len(test_dataset))
    #test_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])
    labels = []
    scores = []
    from tqdm import trange
    for i in trange(len(test_dataset)):
        fact1 = test_dataset[i]["fact_content1"] if test_dataset[i].get('fact1')==None else test_dataset[i]["fact1"]
        fact2 = test_dataset[i]["fact_content2"] if test_dataset[i].get('fact2')==None else test_dataset[i]["fact2"]
        label = test_dataset[i]["label"]
        score = huggingface_finetunedContradictScore(fact1, fact2)
        labels.append(label)
        scores.append(score)
    # Print the average and std and max and min of the scores
    import numpy as np
    print(np.mean(scores), np.std(scores), np.max(scores), np.min(scores))
    from sklearn.metrics import roc_auc_score
    print(roc_auc_score(labels, scores))
    label_pred = [1 if score > 0.5 else 0 for score in scores]
    from sklearn.metrics import accuracy_score, confusion_matrix
    #print(accuracy_score(labels, label_pred))
    print(confusion_matrix(labels, label_pred))
    return labels, scores
    # TODO: Show badcases,
    #  visualize the top 3 highest nli score when the label is 0
    from operator import itemgetter
    print("--------------------------------------------------")
    badcases = []
    for i in range(len(test_dataset)):
        if labels[i] == 0:
            badcases.append((i, scores[i], test_dataset[i]["fact1"], test_dataset[i]["fact2"]))
    badcases = sorted(badcases, key=itemgetter(1), reverse=True)
    print(badcases[:3])
    #  visualize the top 3 lowest nli score when the label is 1
    print("--------------------------------------------------")
    badcases = []
    for i in range(len(test_dataset)):
        if labels[i] == 1:
            badcases.append((i, scores[i], test_dataset[i]["fact1"], test_dataset[i]["fact2"]))
    badcases = sorted(badcases, key=itemgetter(1), reverse=False)
    print(badcases[:3])
    return labels, scores


if __name__ == "__main__":
    #train_model()
    evaluate_model()