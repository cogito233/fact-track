import pandas as pd
import os
BASE_PATH = os.environ["BASE_PATH"]
import sys
sys.path.append(f"{BASE_PATH}/fact-track/fact-track/utils")
from gpt_api import non_determistic_simple_API
def inference_perference_odds(plotx1, plotx2, ploty1, ploty2):
    # print("plotx1: ", plotx1)
    # print("plotx2: ", plotx2)
    # print("#"*50)
    # print("ploty1: ", ploty1)
    # print("ploty2: ", ploty2)
    # exit(0)
    is_swap = False
    import random
    if random.random() > 0.5:
        # plotx1, plotx2 = plotx2, plotx1
        # ploty1, ploty2 = ploty2, ploty1
        plotx1, ploty1 = ploty1, plotx1
        plotx2, ploty2 = ploty2, plotx2
        is_swap = True
    prompt = f"""Pair 1:
plot 1: {plotx1}
plot 2: {plotx2}

Pair 2:
plot 1: {ploty1}
plot 2: {ploty2}

Which one is more likely to be contradict? Reasoning first and then answer "Pair 1" or "Pair 2". Please use the following format:

Reason: [TODO]
Answer: [Pair 1/Pair 2]
"""
    print("Prompt: ", prompt)
    outputs = non_determistic_simple_API('gpt-4-1106-preview', prompt, temp = 0)
    print("Outputs: ", outputs)
    outputs = outputs.split("Answer: ")[1]
    # exit(0)
    if "pair 1" in outputs.lower():
        if is_swap:
            return 1
        else:
            return 0
    elif "pair 2" in outputs.lower():
        if is_swap:
            return 0
        else:
            return 1

def compute_perference_odds(badcase_path):
    df = pd.read_csv(badcase_path)
    df_tp = df[df['type'] == "tp"]
    df_fp = df[df['type'] == "fp"]
    df_fn = df[df['type'] == "fn"]
    print("tp: ", len(df_tp))
    print("fp: ", len(df_fp))
    print("fn: ", len(df_fn))
    # set random seed
    import numpy as np
    np.random.seed(42)
    if len(df_fn) > len(df_fp):
        # subsample fn to the same size as fp
        df_fn = df_fn.sample(n=len(df_fp), random_state=1)
    else:
        # subsample fp to the same size as fn, keep the top max_nli
        # random shuffle
        # df_fp = df_fp.sample(n=len(df_fn), random_state=1)
        # sort by nli
        df_fp = df_fp.sort_values(by=['max_nli'], ascending=False)
        df_fp = df_fp.iloc[:len(df_fn)]
        # print("df_fp: ", df_fp.head())
        # exit(0)
    print("tp: ", len(df_tp))
    print("fp: ", len(df_fp))
    print("fn: ", len(df_fn))
    # random shuffle
    df_fn = df_fn.sample(frac=1, random_state=1)
    df_fp = df_fp.sample(frac=1, random_state=1)
    perference_list = []
    from tqdm import trange
    for i in trange(len(df_fn)):
        plotx1 = df_fn.iloc[i]['plot_1']
        plotx2 = df_fn.iloc[i]['plot_2']
        ploty1 = df_fp.iloc[i]['plot_1'] # 我们认为fp更positive? 答案越大越好
        ploty2 = df_fp.iloc[i]['plot_2']
        perference_list.append(inference_perference_odds(plotx1, plotx2, ploty1, ploty2))
    print("perference_list: ", perference_list)
    print("mean: ", np.mean(perference_list))
    print("len(perference_list): ", len(perference_list))

if __name__ == '__main__':
    #badcase_path = f"{BASE_PATH}/fact-track/dataPostAnno/101x_pure_simple_llama2-7B_block=0.8/badcase_llamaPairwiseAnno.csv"
    badcase_path = f"{BASE_PATH}/fact-track/dataPostAnno/101x_pure_simple_llama2-7B_block=0.8/badcase.csv"
    compute_perference_odds(badcase_path)