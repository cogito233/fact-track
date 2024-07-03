import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from contradictRecog_baseline import *

def load_model(model_name):
    if model_name == "gpt":
        return load_gpt()
    elif model_name == "llama":
        return load_llama()
    else:
        raise ValueError("model_name should be 'gpt' or 'llama'")

def reCheck_factPair(path, model_name="gpt", method="zeroshot", first_half=True):
    if first_half:
        suffix = "_first_half"
    else:
        suffix = "_last_half"
    df_fact = pd.read_csv(os.path.join(path, "fact.csv"))
    dict_fact = df_fact.set_index('fact_key')['fact_content'].to_dict()

    df_factPair = pd.read_csv(os.path.join(path, "fact_contradict.csv"))
    model = load_model(model_name)

    if method.lower() == "zeroshot":
        method_function = unitContradictCheck
    elif method.lower() == "fewshot":
        method_function = fewshotContradictCheck
    else:
        raise ValueError("method should be 'zeroshot' or 'fewshot'")

    # 使用 itertuples 优化迭代
    i = 0
    mid = len(df_factPair) // 2
    for row in tqdm(df_factPair.itertuples()):
        if first_half and i >= mid:
            break
        if not first_half and i < mid:
            i += 1
            continue
        i += 1
        fact1_content = dict_fact[row.fact1_key]
        fact2_content = dict_fact[row.fact2_key]
        contradict = method_function(fact1_content, fact2_content, model)
        df_factPair.at[row.Index, 'contradict'] = contradict  # 使用 `at` 进行赋值

        # 每100次迭代保存一次数据
        if row.Index % 100 == 99:
            df_factPair.to_csv(os.path.join(path, f"fact_contradict_{model_name}_{method}{suffix}.csv"), index=False)
            # exit(0)

    df_factPair.to_csv(os.path.join(path, f"fact_contradict_{model_name}_{method}{suffix}.csv"), index=False)

if __name__ == "__main__":
    path = "/home/yangk/zhiheng/fact-track/dataPostAnno/1100_1200_pure_simple_detect_llama2-7B"
    model = "gpt"
    method = "fewshot"
    reCheck_factPair(path, model, method, first_half=True)