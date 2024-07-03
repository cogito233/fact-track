import pandas as pd
import os
BASE_PATH = os.environ["BASE_PATH"]

import sys
sys.path.append(f"{BASE_PATH}/fact-track/fact-track/utils")
from gpt_api import non_determistic_simple_API

def inference_score_simple(plot1, plot2):
    prompt = f"""{plot1}
{plot2}

Do these two given plot in a story contain plot redundant or factual inconsistency (they are assumed to be happenn on different stage in story since the index is not overlap)? Simply score the plot's redundant and factual inconsistency level using one score from 1 (lowest) to 5 (highest).

Please simpliy answer:

Score of Redundancy and Factual Inconsistency: [TODO, a number from 1 to 5]
"""
    # print("Prompt: ", prompt)
    # outputs = non_determistic_simple_API('gpt-4-1106-preview', prompt, temp=0)
    outputs = non_determistic_simple_API('gpt-4-0125-preview', prompt, temp=0)
    print("Outputs: ", outputs)
    return int(outputs.split("\n")[0].split(":")[1].strip())

def inference_score_outline(outline_id, plot1, plot2):
    # Based on the full outline context and do the scoring
    from oddAnno import load_full_outline_text
    outline_text = load_full_outline_text(outline_id)
    prompt = f"""Consider the following story outline written by a ai assistant, the outline follows a tree structure, for example, the son of node 1 is 1.1, 1.2, 1.3 respectively. The story outline is as follows:

{outline_text}


Consider the following two plots in the given story outline:
{plot1}
{plot2}

Do these two given plot in a story contain plot redundant or factual inconsistency (they are assumed to be happenn on different stage in story since the index is not overlap)? Simply score the plot's redundant and factual inconsistency level using one score from 1 (lowest) to 5 (highest).

Please simpliy answer:

Score of Redundancy and Factual Inconsistency: [TODO, a number from 1 to 5]
"""
    # print("Prompt: ", prompt)
    # outputs = non_determistic_simple_API('gpt-4-1106-preview', prompt, temp=0)
    outputs = non_determistic_simple_API('gpt-4-0125-preview', prompt, temp=0)
    print("Outputs: ", outputs)
    return int(outputs.split("\n")[0].split(":")[1].strip())

def scoreAnno(result_dir, suffix="max_nli_gpt4Anno&llamaDect.csv", is_simple=False):
    match_df_path = os.path.join(result_dir, "oddExp", f"oddMatch_{suffix}")
    match_df = pd.read_csv(match_df_path)

    result_path = os.path.join(result_dir, "oddExp", f"scoreAnno_{'simple_' if is_simple else ''}{suffix}")
    if os.path.exists(result_path):
        print(f"File {result_path} already exists!")
        return

    results = []
    from tqdm import tqdm
    for index, row in tqdm(match_df.iterrows()):
        score_1 = score_2 = 0  # Default scores
        if is_simple:
            try:
                score_1 = inference_score_simple(row['plot_1A'], row['plot_1B'])
            except:
                pass
            try:
                score_2 = inference_score_simple(row['plot_2A'], row['plot_2B'])
            except:
                pass
        else:
            score_1 = inference_score_outline(row['outline_id'], row['plot_1A'], row['plot_1B'])
            score_2 = inference_score_outline(row['outline_id'], row['plot_2A'], row['plot_2B'])

        results.append({
            'outline_id': row['outline_id'],
            'plot_1A': row['plot_1A'],
            'plot_1B': row['plot_1B'],
            'type_1': row['type_1'],
            'plot_2A': row['plot_2A'],
            'plot_2B': row['plot_2B'],
            'type_2': row['type_2'],
            'score_1': score_1,
            'score_2': score_2
        })

    pd.DataFrame(results).to_csv(result_path, index=False)

def scoreAnalyze(result_dir, suffix = "max_nli_gpt4Anno&llamaDect.csv", is_simple = False):
    if is_simple:
        result_path = f"{result_dir}/oddExp/scoreAnno_simple_{suffix}"
    else:
        result_path = f"{result_dir}/oddExp/scoreAnno_{suffix}"
    result_df = pd.read_csv(result_path)
    print("Total: ", len(result_df))
    print("Score 1", sum(result_df['score_1'])/sum(result_df['score_1']!=0))
    print("Score 2", sum(result_df['score_2'])/sum(result_df['score_2']!=0))

def pairwise_pipeline(result_dir):
    suffix = "llamaPairwise&llamaDect.csv"
    result_name = f"badcase_{suffix}"
    if os.path.exists(f"{result_dir}/{result_name}"):
        print("Badcase file already exists!")
    else:
        sys.path.append(f"{BASE_PATH}/fact-track/fact-track/autoAnno")
        from postAnno_badcaseAnalyze import simplify_save_badcase
        simplify_save_badcase(path = result_dir,
                     anno_path = "plot_contradict_pred.csv",
                     pred_path = "plot_contradict_baseline_llama2-7B-chat.csv",
                     output_filename = result_name)
    from oddMatch import oddMatch
    oddMatch(result_dir, suffix = suffix, strategy = "random")
    # exit(0)
    scoreAnno(result_dir, "random_" + suffix, is_simple=True)
    scoreAnalyze(result_dir, "random_" + suffix, is_simple=True)

if __name__ == "__main__":
    # suffix = "random_random&gpt4Anno.csv"

    suffix = "max_nli_gpt4Anno&llamaDect.csv"
    result_dir = f"{BASE_PATH}/fact-track/dataPostAnno/1100_1200_pure_simple_detect_llama2-7B"
    # #scoreAnno(result_dir, suffix, is_simple=True)
    scoreAnno(result_dir, suffix, is_simple=False)
    scoreAnalyze(result_dir, suffix, is_simple=False)
    # pairwise_pipeline(result_dir)