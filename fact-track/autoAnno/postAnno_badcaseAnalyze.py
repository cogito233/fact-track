import os
BASE_PATH = os.environ["BASE_PATH"]
import pandas as pd

import sys
sys.path.append(f"{BASE_PATH}/fact-track/fact-track/utils")
from huggingface_api import huggingface_UnitContradictScore
def print_FP_to_file(plot1_id, plot2_id, outline_id, plot_df, fact_df, fact_contradict_df, filename):
    max_nli = -1
    with open(filename, 'a') as file:
        file.write("#" * 100 + "\n")
        file.write(f"FP sample, plot1_id: {plot1_id}, plot2_id: {plot2_id}, outline_id: {outline_id}\n")
        plot1 = plot_df[(plot_df['plot_id'] == plot1_id) & (plot_df['outline_id'] == outline_id)]['plot_content'].values[0]
        plot2 = plot_df[(plot_df['plot_id'] == plot2_id) & (plot_df['outline_id'] == outline_id)]['plot_content'].values[0]
        file.write(f"plot1: {plot1}\n")
        file.write(f"plot2: {plot2}\n")

        fact_df = fact_df[fact_df['outline_id'] == outline_id]
        fact_plot1_df = fact_df[fact_df['plot_id'] == plot1_id]
        fact_plot1_df = fact_plot1_df[fact_plot1_df['fact_type'] == "postfact"]
        fact_plot2_df = fact_df[fact_df['plot_id'] == plot2_id]
        fact_plot2_df = fact_plot2_df[fact_plot2_df['fact_type'] == "prefact"]

        for i in range(len(fact_plot1_df)):
            fact_key1 = fact_plot1_df.iloc[i]['fact_key']
            content_1 = fact_plot1_df.iloc[i]['fact_content']
            for j in range(len(fact_plot2_df)):
                fact_key2 = fact_plot2_df.iloc[j]['fact_key']
                content_2 = fact_plot2_df.iloc[j]['fact_content']
                fact_contradict_df_sub = fact_contradict_df[fact_contradict_df['fact1_key'] == fact_key1]
                fact_contradict_df_sub = fact_contradict_df_sub[fact_contradict_df_sub['fact2_key'] == fact_key2]
                if len(fact_contradict_df_sub) > 0:
                    fact_contradict = fact_contradict_df_sub.iloc[0]
                    if fact_contradict['contradict']:
                        file.write(f"fact1: {content_1}, fact2: {content_2}, fact1_key: {fact_key1}, fact2_key: {fact_key2}, nli_score: {fact_contradict['nli_score']}\n")
                        if fact_contradict['nli_score'] > max_nli:
                            max_nli = fact_contradict['nli_score']
        file.write("\n")
    return max_nli

def print_FN_to_file(plot1_id, plot2_id, outline_id, plot_df, fact_df, fact_contradict_df, filename):
    max_nli = -1
    with open(filename, 'a') as file:
        file.write("#" * 100 + "\n")
        file.write(f"FN sample, plot1_id: {plot1_id}, plot2_id: {plot2_id}, outline_id: {outline_id}\n")
        plot1 = plot_df[(plot_df['plot_id'] == plot1_id) & (plot_df['outline_id'] == outline_id)]['plot_content'].values[0]
        plot2 = plot_df[(plot_df['plot_id'] == plot2_id) & (plot_df['outline_id'] == outline_id)]['plot_content'].values[0]
        file.write(f"plot1: {plot1}\n")
        file.write(f"plot2: {plot2}\n")

        fact_df = fact_df[fact_df['outline_id'] == outline_id]
        fact_plot1_df = fact_df[fact_df['plot_id'] == plot1_id]
        fact_plot1_df = fact_plot1_df[fact_plot1_df['fact_type'] == "postfact"]
        fact_plot2_df = fact_df[fact_df['plot_id'] == plot2_id]
        fact_plot2_df = fact_plot2_df[fact_plot2_df['fact_type'] == "prefact"]
        fact_pair_list = []

        for i in range(len(fact_plot1_df)):
            content_1 = fact_plot1_df.iloc[i]['fact_content']
            l1 = fact_plot1_df.iloc[i]['l']
            r1 = fact_plot1_df.iloc[i]['r']
            for j in range(len(fact_plot2_df)):
                content_2 = fact_plot2_df.iloc[j]['fact_content']
                l2 = fact_plot2_df.iloc[j]['l']
                r2 = fact_plot2_df.iloc[j]['r']
                score = huggingface_UnitContradictScore(content_1, content_2)
                status = "partially overlap"
                if l2 < l1 and l1 < r2 and r2 < r1:
                    status = "overlap"
                elif l1 < r1 and r1 < l2 and l2 < r2:
                    status = "blocked"
                fact_pair_list.append((content_1, content_2, score, status))

        fact_pair_list = sorted(fact_pair_list, key=lambda x: x[2], reverse=True)
        file.write(f"Top 5 contradict fact pair:\n")
        for i in range(min(5, len(fact_pair_list))):
            file.write(f"fact1: {fact_pair_list[i][0]}, fact2: {fact_pair_list[i][1]}, score: {fact_pair_list[i][2]}, status: {fact_pair_list[i][3]}\n")
            if fact_pair_list[i][2] > max_nli:
                max_nli = fact_pair_list[i][2]
        file.write("\n")
    return max_nli

def print_badcase(path = "/home/yangk/zhiheng/develop_codeversion/fact-track/dataPostAnno/101x_pure_simple",
                  anno_path = None):
    if anno_path is None:
        anno_path = os.path.join(path, "plot_contradict_anno.csv")
    pred_path = os.path.join(path, "plot_contradict_pred.csv")
    # Output filename
    output_path = os.path.join(path, "badcase.txt")
    anno_df = pd.read_csv(anno_path)
    pred_df = pd.read_csv(pred_path)

    # Now is load the basic statistics
    plot_path = os.path.join(path, "plot.csv")
    fact_path = os.path.join(path, "fact.csv")
    fact_contradict_path = os.path.join(path, "fact_contradict.csv")
    plot_df = pd.read_csv(plot_path)
    fact_df = pd.read_csv(fact_path)
    fact_contradict_df = pd.read_csv(fact_contradict_path)
    fp, fn = 0, 0
    tp, tn = 0, 0
    fp_nli, fn_nli = [], []
    tp_nli = []
    # if already exist, delete it
    if os.path.exists(output_path):
        os.remove(output_path)
    from tqdm import trange
    for i in trange(len(pred_df)):
        label = pred_df.iloc[i]['type_byPred']
        gt_label = anno_df.iloc[i]['type_byAnno']
        plot1_id = anno_df.iloc[i]['plot1_id']
        plot2_id = anno_df.iloc[i]['plot2_id']
        outline_id = anno_df.iloc[i]['outline_id']
        if gt_label == "contradict":
            if label == "contradict":
                tp += 1
                # nli = print_FP_to_file(plot1_id, plot2_id, outline_id, plot_df, fact_df, fact_contradict_df, output_path)
                # tp_nli.append(nli)
            else:
                fn += 1
                nli = print_FN_to_file(plot1_id, plot2_id, outline_id, plot_df, fact_df, fact_contradict_df, output_path)
                fn_nli.append(nli)
        else:
            if label == "contradict":
                fp += 1
                nli = print_FP_to_file(plot1_id, plot2_id, outline_id, plot_df, fact_df, fact_contradict_df, output_path)
                fp_nli.append(nli)
            elif label != "not sampled":
                tn += 1
    print(f"tp: {tp}, tf: {tn}")
    print(f"fp: {fp}, fn: {fn}")
    print(f"fp_nli: {fp_nli}")
    print(f"fn_nli: {fn_nli}")
    print(f"tp_nli: {tp_nli}")


def get_maxNLI_inFact(plot1_id, plot2_id, outline_id, plot_df, fact_df, fact_contradict_df):
    max_nli = -1
    fact_df = fact_df[fact_df['outline_id'] == outline_id]
    fact_plot1_df = fact_df[fact_df['plot_id'] == plot1_id]
    fact_plot1_df = fact_plot1_df[fact_plot1_df['fact_type'] == "postfact"]
    fact_plot2_df = fact_df[fact_df['plot_id'] == plot2_id]
    fact_plot2_df = fact_plot2_df[fact_plot2_df['fact_type'] == "prefact"]
    for i in range(len(fact_plot1_df)):
        fact_key1 = fact_plot1_df.iloc[i]['fact_key']
        for j in range(len(fact_plot2_df)):
                fact_key2 = fact_plot2_df.iloc[j]['fact_key']
                fact_contradict_df_sub = fact_contradict_df[fact_contradict_df['fact1_key'] == fact_key1]
                fact_contradict_df_sub = fact_contradict_df_sub[fact_contradict_df_sub['fact2_key'] == fact_key2]
                if len(fact_contradict_df_sub) > 0:
                    fact_contradict = fact_contradict_df_sub.iloc[0]
                    if fact_contradict['contradict']:
                        if fact_contradict['nli_score'] > max_nli:
                            max_nli = fact_contradict['nli_score']
    return max_nli

# Used to estimate the odds
def save_badcase(path = "/home/yangk/zhiheng/develop_codeversion/fact-track/dataPostAnno/101x_pure_simple", anno_path = None, pred_path = None, output_filename = "badcase_gptAnno&llamaDect.csv", isDetect = True): # If isDetect is False, then let NLI always be 0
    if anno_path is None:
        anno_path = os.path.join(path, "plot_contradict_anno.csv")
    if pred_path is None:
        pred_path = os.path.join(path, "plot_contradict_pred.csv")
    # pred_path = "/home/yangk/zhiheng/develop_codeversion/fact-track/dataPostAnno/101x_pure_simple/plot_contradict_baseline.csv"
    # Output filename
    # output_path = os.path.join(path, "badcase_llamaPairwiseAnno.csv")
    output_path = os.path.join(path, output_filename)
    print("anno_path: ", anno_path)
    print("pred_path: ", pred_path)
    anno_df = pd.read_csv(anno_path)
    pred_df = pd.read_csv(pred_path)
    anno_column = "type_byAnno" if "type_byAnno" in anno_df.columns else "type_byPred"
    pred_column = "type_byPred" if "type_byPred" in pred_df.columns else "type_byAnno"
    # Now is load the basic statistics
    plot_path = os.path.join(path, "plot.csv") # plot_id,outline_id -> plot_content
    plot_df = pd.read_csv(plot_path)
    plot_dict = {}
    for i in range(len(plot_df)):
        plot_dict[(plot_df.iloc[i]['plot_id'], plot_df.iloc[i]['outline_id'])] = plot_df.iloc[i]['plot_content']

    fact_path = os.path.join(path, "fact.csv")
    fact_contradict_path = os.path.join(path, "fact_contradict.csv")
    fact_df = pd.read_csv(fact_path)
    fact_contradict_df = pd.read_csv(fact_contradict_path)

    fp, fn = 0, 0
    tp, tn = 0, 0
    # if already exist, delete it
    if os.path.exists(output_path):
        os.remove(output_path)
    from tqdm import trange
    result_dict = {
        "outline_id": [],
        "plot_1": [],
        "plot_2": [],
        "type": [], # fp or fn or tp, do not consider tn
        "max_nli": [], # only work for tp and fp
    }
    for i in trange(len(pred_df)):
        label = pred_df.iloc[i][pred_column]
        gt_label = anno_df.iloc[i][anno_column]
        plot1_id = anno_df.iloc[i]['plot1_id']
        plot2_id = anno_df.iloc[i]['plot2_id']
        outline_id = anno_df.iloc[i]['outline_id']
        plot1_content = plot_dict[(plot1_id, outline_id)]
        plot2_content = plot_dict[(plot2_id, outline_id)]
        if gt_label == "contradict":
            if label == "contradict":
                tp += 1
                result_dict["outline_id"].append(outline_id)
                result_dict["plot_1"].append(plot1_content)
                result_dict["plot_2"].append(plot2_content)
                result_dict["type"].append("tp")
                if isDetect:
                    result_dict["max_nli"].append(get_maxNLI_inFact(plot1_id, plot2_id, outline_id, plot_df, fact_df, fact_contradict_df))
                else:
                    result_dict["max_nli"].append(0)
            else:
                fn += 1
                result_dict["outline_id"].append(outline_id)
                result_dict["plot_1"].append(plot1_content)
                result_dict["plot_2"].append(plot2_content)
                result_dict["type"].append("fn")
                result_dict["max_nli"].append(0)
                # 为什么fn用0? fp是提高阈值变成tn，fn是gpt说好 但我们觉得不好
        else:
            if label == "contradict":
                fp += 1
                result_dict["outline_id"].append(outline_id)
                result_dict["plot_1"].append(plot1_content)
                result_dict["plot_2"].append(plot2_content)
                result_dict["type"].append("fp")
                if isDetect:
                    result_dict["max_nli"].append(get_maxNLI_inFact(plot1_id, plot2_id, outline_id, plot_df, fact_df, fact_contradict_df))
                else:
                    result_dict["max_nli"].append(0)
            elif label != "not":
                tn += 1
    print(f"tp: {tp}, tf: {tn}")
    print(f"fp: {fp}, fn: {fn}")
    result_df = pd.DataFrame(result_dict)
    result_df.to_csv(output_path, index=False)



def simplify_save_badcase(path, output_filename="badcase_gptAnno&llamaDect.csv", anno_path = "plot_contradict_anno.csv", pred_path = "plot_contradict_pred.csv"):
    anno_path = os.path.join(path, anno_path)
    pred_path = os.path.join(path, pred_path)
    plot_path = os.path.join(path, "plot.csv")
    # fact_path = os.path.join(path, "fact.csv")
    # fact_contradict_path = os.path.join(path, "fact_contradict.csv")
    output_path = os.path.join(path, output_filename)

    # Load data
    anno_df = pd.read_csv(anno_path)
    pred_df = pd.read_csv(pred_path)
    plot_df = pd.read_csv(plot_path)
    # fact_df = pd.read_csv(fact_path)
    # fact_contradict_df = pd.read_csv(fact_contradict_path)

    # Create plot dictionary
    plot_dict = {(row['plot_id'], row['outline_id']): row['plot_content'] for index, row in plot_df.iterrows()}

    # Initialize counters and result storage
    counts = {"fp": 0, "fn": 0, "tp": 0, "tn": 0}
    result_dict = {"outline_id": [], "plot_1": [], "plot_2": [], "type": [], "max_nli": []}

    # Determine correct column names
    anno_column = "type_byAnno" if "type_byAnno" in anno_df.columns else (
        "type_byPred" if "type_byPred" in anno_df.columns else "type_byBaseline")
    pred_column = "type_byPred" if "type_byPred" in pred_df.columns else (
        "type_byAnno" if "type_byAnno" in pred_df.columns else "type_byBaseline")

    # Process predictions
    for _, row in pred_df.iterrows():
        label, gt_label = row[pred_column], anno_df.loc[_, anno_column]
        plot1_id, plot2_id, outline_id = row['plot1_id'], row['plot2_id'], row['outline_id']
        plot1_content, plot2_content = plot_dict[(plot1_id, outline_id)], plot_dict[(plot2_id, outline_id)]
        type_label = "fn" if gt_label == "contradict" and label != "contradict" else "fp" if label == "contradict" and gt_label != "contradict" else "tp" if label == "contradict" and gt_label == "contradict" else "tn"
        counts[type_label] += 1
        if type_label in ["fp", "fn", "tp"]:
            result_dict["outline_id"].append(outline_id)
            result_dict["plot_1"].append(plot1_content)
            result_dict["plot_2"].append(plot2_content)
            result_dict["type"].append(type_label)
            max_nli = 0
            result_dict["max_nli"].append(max_nli)

    print(f"tp: {counts['tp']}, tn: {counts['tn']}")
    print(f"fp: {counts['fp']}, fn: {counts['fn']}")

    # Save results to CSV
    pd.DataFrame(result_dict).to_csv(output_path, index=False)

if __name__ == "__main__":
    pred_path = "/home/yangk/zhiheng/develop_codeversion/fact-track/dataPostAnno/101x_pure_simple_llama2-7B_block=0.8_new/"
    anno_path = "/home/yangk/zhiheng/develop_codeversion/fact-track/dataPostAnno/101x_pure_simple/plot_contradict_anno.csv"
    print_badcase(pred_path, anno_path)
    # save_badcase(pred_path, anno_path)