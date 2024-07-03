import pandas as pd
import os
BASE_PATH = os.environ["BASE_PATH"]

import sys
sys.path.append(f"{BASE_PATH}/fact-track/fact-track/utils")
from gpt_api import non_determistic_simple_API

sys.path.append(f"{BASE_PATH}/fact-track/fact-track/core")
from outline import Outline, OutlineItem

sys.path.append(f"{BASE_PATH}/fact-track/fact-track/analyze")
from convert_dataTemplate import outline2text_full
# TODO: currently we directly convert from html format, in future we should put outline directly in our database

def inference_perference_simple(plotx1, plotx2, ploty1, ploty2):
    # print("plotx1: ", plotx1)
    # print("plotx2: ", plotx2)
    # print("#"*50)
    # print("ploty1: ", ploty1)
    # print("ploty2: ", ploty2)
    # exit(0)
    is_swap = False
    import random
    if random.random() > 0.5:
        plotx1, ploty1 = ploty1, plotx1
        plotx2, ploty2 = ploty2, plotx2
        is_swap = True
    prompt = f"""Pair 1:
plot 1: {plotx1}
plot 2: {plotx2}

Pair 2:
plot 1: {ploty1}
plot 2: {ploty2}

Which one is more likely to be contradict? Only answer "Pair 1" or "Pair 2".
"""
    print("Prompt: ", prompt)
    outputs = non_determistic_simple_API('gpt-4-1106-preview', prompt, temp = 0)
    print("Outputs: ", outputs)
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

def load_full_outline_text(outline_id):
    def remove_html_and_replace_click_to_fold(text):
        import re
        # Define the regular expression for HTML tags
        html_tag_pattern = re.compile(r'<[^>]+>')
        text = text.replace('<div class="depth_3">', '\n')
        # Replace HTML tags with an empty string
        text_without_html = html_tag_pattern.sub('', text)

        # Replace '[click to fold]' with a newline character
        text_with_replacements = text_without_html.replace('[click to fold]', '\n')

        # Replace '<div class="depth_3">' with a newline character
        return text_with_replacements
    outline_path = f"{BASE_PATH}/fact-track/data/{outline_id}_llama2-7B/object/outline.pkl"
    import pickle
    with open(outline_path, "rb") as f:
        outline = pickle.load(f)
    outline_text = outline2text_full(outline)
    outline_text = remove_html_and_replace_click_to_fold(outline_text)
    # print("Outline text: ", outline_text)
    return outline_text

def inference_perference_outline(outline_id, plotx1, plotx2, ploty1, ploty2):
    # print("outline_id: ", outline_id)
    # print("#"*50)
    # print("plotx1: ", plotx1)
    # print("plotx2: ", plotx2)
    # print("#"*50)
    # print("ploty1: ", ploty1)
    # print("ploty2: ", ploty2)
    # print("#"*50)
    outline_text = load_full_outline_text(outline_id)
    # print("outline_text: ", outline_text)
    # exit(0)
    import random
    is_swap = False
    if random.random() > 0.5:
        plotx1, ploty1 = ploty1, plotx1
        plotx2, ploty2 = ploty2, plotx2
        is_swap = True
    prompt = f"""In this task we will give you a story outline, and two plot pairs select in this story outline, your task is to identify which plot pair is more likely to be contradict with each other based on the given outline.
Please take into account that the outline is structured as a tree. In this tree-like structure, individual points such as 1.1, 1.2, and 1.3 are child nodes of plot point 1, so there is no contradiction between a node with its ancestors such as 1.3 and 1. 
Outline:
{outline_text}

Pair 1:
plot 1: {plotx1}
plot 2: {plotx2}

Pair 2:
plot 1: {ploty1}
plot 2: {ploty2}

Which plot pair is more likely to be contradict with each other? Only answer "Pair 1" or "Pair 2".
"""

    outputs = non_determistic_simple_API('gpt-4-1106-preview', prompt, temp = 0)
    # print("Prompt: ", prompt)
    # print("Outputs: ", outputs)
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

    # return is a value in [0, 1]
    # 0 means pair 1 is more likely to be contradict


def oddAnno(result_dir, suffix = "max_nli_gpt4Anno&llamaDect.csv", is_simple = False):
    match_df = pd.read_csv(f"{result_dir}/oddExp/oddMatch_{suffix}")
    if is_simple:
        result_path = f"{result_dir}/oddExp/oddAnno_simple_{suffix}"
    else:
        result_path = f"{result_dir}/oddExp/oddAnno_{suffix}"
    if os.path.exists(result_path):
        print(f"File {result_path} already exists!")
        return
    # outline_id, plot_1A, plot_1B, type_1, max_nli_1, plot_2A, plot_2B, type_2, max_nli_2
    results = []
    import random
    random.seed(42)
    # 遍历 match_df 处理每一行
    i = 0
    from tqdm import tqdm
    for index, row in tqdm(match_df.iterrows(), total=match_df.shape[0]):
        # 调用 inference_perference_outline 函数
        # if random.random() > 0.5:
        if is_simple:
            result = inference_perference_simple(row['plot_1A'], row['plot_1B'], row['plot_2A'], row['plot_2B'])
        else:
            result = inference_perference_outline(row['outline_id'], row['plot_1A'], row['plot_1B'], row['plot_2A'], row['plot_2B'])
        # else:
        #     result = 1 - inference_perference_outline(row['outline_id'], row['plot_2A'], row['plot_2B'], row['plot_1A'], row['plot_1B'])
        # 添加结果到列表
        results.append({
            'outline_id': row['outline_id'],
            'plot_1A': row['plot_1A'],
            'plot_1B': row['plot_1B'],
            'type_1': row['type_1'],
            'plot_2A': row['plot_2A'],
            'plot_2B': row['plot_2B'],
            'type_2': row['type_2'],
            'result': result
        })
        i += 1
        if i % 100 == 0:
            # Save the partial result by outline_id, plot_1A, plot_1B, type_1, plot_2A, plot_2B, type_2, result
            result_df = pd.DataFrame(results)
            result_df.to_csv(f"{result_dir}/oddAnno_{i}_{suffix}", index=False)

    # Save the result by outline_id, plot_1A, plot_1B, type_1, plot_2A, plot_2B, type_2, result
    # 转换结果列表为 DataFrame
    result_df = pd.DataFrame(results)
    # 保存 DataFrame 到 CSV
    result_df.to_csv(result_path, index=False)

def oddAnalyze(result_dir, suffix = "max_nli_gpt4Anno&llamaDect.csv", is_simple = False):
    if is_simple:
        result_path = f"{result_dir}/oddExp/oddAnno_simple_{suffix}"
    else:
        result_path = f"{result_dir}/oddExp/oddAnno_{suffix}"
    result_df = pd.read_csv(result_path)
    print("Total: ", len(result_df))
    print("FP>FN: ", len(result_df[result_df['result'] == 0]))
    print("Ratio: ", len(result_df[result_df['result'] == 0]) / len(result_df))

def oddExp_pipeline(source = "random", target = "gpt4Anno", is_simple = False, strategy = "random", result_dir = None): # Method work only if under our method
    if source == "random":
        source_name = "plot_contradict_random.csv"
    elif source == "llamaPair":
        source_name = "plot_contradict_baseline_llama2-7B-chat.csv"
    elif source == "llamaDect":
        source_name = "plot_contradict_baseline_pred.csv"
    else:
        raise ValueError("Invalid source")
    if target == "gpt4Anno":
        target_name = "plot_contradict_anno.csv"
    else:
        raise ValueError("Invalid target")
    result_name = f"badcase_{source}&{target}.csv"
    if os.path.exists(f"{result_dir}/{result_name}"):
        print("Badcase file already exists!")
    else:
        sys.path.append(f"{BASE_PATH}/fact-track/fact-track/autoAnno")
        from postAnno_badcaseAnalyze import save_badcase
        save_badcase(path = result_dir,
                     anno_path = os.path.join(result_dir, source_name),
                     pred_path = os.path.join(result_dir, target_name),
                     output_filename = result_name,
                     isDetect = True if "Dect" in target else False)
    # Now we have the badcase, we can use the badcase to do the oddAnno
    from oddMatch import oddMatch
    oddMatch(result_dir, suffix = f"{source}&{target}.csv", strategy = strategy)
    oddAnno(result_dir, suffix = f"{strategy}_{source}&{target}.csv", is_simple = is_simple)
    oddAnalyze(result_dir, suffix = f"{strategy}_{source}&{target}.csv", is_simple = is_simple)

if __name__ == "__main__":
    result_dir = f"{BASE_PATH}/fact-track/dataPostAnno/1100_1200_pure_simple_detect_llama2-7B"
    oddExp_pipeline(source = "random", target = "gpt4Anno", is_simple = True, strategy = "random", result_dir = result_dir)