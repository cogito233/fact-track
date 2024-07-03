import os
BASE_PATH = os.environ["BASE_PATH"]
import sys
sys.path.append(f"{BASE_PATH}/fact-track/fact-track/utils")
from log_saver import LogSaver

sys.path.append(f"{BASE_PATH}/fact-track/fact-track/core")

# Two requirement:
# 1. Export the dataset to annotate
def export_detection_dataset(metaname, outline, path):
    # Return a one-line csv, detection_data.csv and outline.txt
    # Only contain the depth until the failure point? or contain full outline?
    from convert_dataTemplate import outline2text,  outline2text_full
    result_df = {"outline_idx": [], "premise": [], "outline_text": [], "outline_text_full": []}
    outline_text = outline2text(outline)
    outline_text_full = outline2text_full(outline)
    result_df["outline_idx"].append(metaname)
    result_df["premise"].append(outline.premise)
    result_df["outline_text"].append(outline_text)
    result_df["outline_text_full"].append(outline_text_full)
    import pandas as pd
    df = pd.DataFrame(result_df)
    df.to_csv(f"{path}/outline_annotation.csv")
    with open(f"{path}/outline.txt", "w") as f:
        f.write(outline_text)
    with open(f"{path}/outline_full.txt", "w") as f:
        f.write(outline_text_full)

# 2. Export all contradict pairs for human to read.
def export_contradict_pairs(stateChecker_dict, path):
    # For Reference:
    # {"idx", "plot", "error",
    #  "facts":[
    #  {"error fact",
    #  "error fact interval",
    #  "exist fact":[{"fact", "fact interval", "plot", "plot id", isPrefact, nli_score}]
    #  ]}]
    #  }
    # return a txt file contains all contradict pairs

    # For each pair, only keep the most fatal one then it is easy to check
    dict_max = {}
    dict_text = {}

    for idx, stateChecker in stateChecker_dict.items():
        if not stateChecker.observation_dict['error']:
            continue
        curr_dict = stateChecker.observation_dict # This is the plot pair that is already after processing
        for fact in curr_dict['facts']:
            for exist_fact in fact['exist fact']:
                idx1, idx2 = exist_fact['plot id'], curr_dict['idx']
                if idx1 > idx2:
                    idx1, idx2 = idx2, idx1
                if (idx1, idx2) not in dict_max:
                    dict_max[(idx1, idx2)] = -1
                if exist_fact['nli_score'] > dict_max[(idx1, idx2)]:
                    dict_max[(idx1, idx2)] = exist_fact['nli_score']
                    dict_text[(idx1, idx2)] = f"""{curr_dict['idx']}: {curr_dict['plot']}
fact: {fact['error fact']}
{exist_fact['plot id']}: {exist_fact['plot']}
exist fact: {exist_fact['fact']}
nli_score: {exist_fact['nli_score']}
################################################

"""
    with open(f"{path}/contradict_pairs.txt", "w") as f:
        for idx1, idx2 in dict_max:
            f.write(dict_text[(idx1, idx2)])

def outline_analyze(metaname = "sample"):
    path = f"{BASE_PATH}/fact-track/data/{metaname}/analyze_result"
    import os
    if not os.path.exists(path):
        os.makedirs(path)
    logSaver = LogSaver(metaname)
    export_dict = logSaver.load()
    print(export_dict.keys())
    # print(export_dict['model_summary'])
    # print(export_dict['detector'])
    # print(export_dict['outline'])
    # print(export_dict['stateChecker_dict'])
    export_detection_dataset(metaname, export_dict['outline'], path)
    export_contradict_pairs(export_dict['stateChecker_dict'], path)

def concat_dataset():
    # return a dataset with 5 outlines
    pass

if __name__ == "__main__":
    # for i in range(995, 1000):
    #     outline_analyze(f"{i}_pure_simple_detect")
    outline_analyze(f"1210_pure_simple_detect_llama2-7B")
