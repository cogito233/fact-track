import pandas as pd

import os
BASE_PATH = os.environ["BASE_PATH"]

import sys
sys.path.append(f"{BASE_PATH}/fact-track/fact-track/utils")
from gpt_api import load_model2classification, openai_UnitContradictCheck

def zeroShotInference(plot1_text, plot2_text, model, idx1, idx2):
    # if idx1 is start with idx2 or idx2 is start with idx1, then they are invalid
    if  len(idx1) > len(idx2) and idx1[:len(idx2)] == idx2 or \
        len(idx2) > len(idx1) and idx2[:len(idx1)] == idx1:
        return "invalid", "NULL"
    # Return label = {"contradict", "not contradict", "invalid"}, model_output = {"Yes", "No", "NULL"}
    prompt_question = "Question: Does those two time-ordered plot point contradict each other? Answer \"Yes\" or \"No\"."
    prompt = f"""{prompt_question}
Point {plot1_text}
Point {plot2_text}

Answer:"""
    # print(prompt)
    answer = model([prompt])[0]
    label = "contradict" if "Yes" in answer or "yes" in answer else "not contradict"
    # print(answer, label)
    # exit(0)
    return label, answer

# TODO: Generate the meta content for the inference
def inference_given_plot(output_dir, pair_list, model_name = 'gpt-3.5-turbo'):
    # model = 'gpt-4-1106-preview' or 'gpt-3.5-turbo'
    curr_pred_dict = {}
    output_path = os.path.join(output_dir, "plot_contradict_pairwise_baseline.csv")
    if "gpt" in model_name:
        model = load_model2classification(model_name)
    elif model_name == "llama2-7B-chat":
        sys.path.append(f"{BASE_PATH}/fact-track/fact-track/utils")
        from llama_api_vLLM import load_deterministic_llama2
        model = load_deterministic_llama2("7B-chat")
    plot_path = os.path.join(output_dir, "plot.csv")
    plot_df = pd.read_csv(plot_path) # indix by (plot_id,outline_id)
    plot_dict = plot_df.set_index(['plot_id', 'outline_id'])['plot_content'].to_dict()
    # print(plot_dict)
    from tqdm import tqdm
    for plot1_id, plot2_id, outline_id in tqdm(pair_list): # this is all the pairs we want to inference
        if (plot1_id, plot2_id, outline_id) in curr_pred_dict:
            continue
        plot1_text = plot_dict[(plot1_id, outline_id)]
        plot2_text = plot_dict[(plot2_id, outline_id)]
        # print("########"*50)
        # print(f"plot1_id: {plot1_id}, plot2_id: {plot2_id}, outline_id: {outline_id}")
        # print(f"plot1_text: {plot1_text}")
        # print(f"plot2_text: {plot2_text}")
        # print("########"*50)
        label, model_output = zeroShotInference(plot1_text, plot2_text, model, plot1_id, plot2_id)
        curr_pred_dict[(plot1_id, plot2_id, outline_id)] = label
        # print(f"plot1_id: {plot1_id}, plot2_id: {plot2_id}, outline_id: {outline_id}, label: {label}, model_output: {model_output}")
    return curr_pred_dict

def baseline_inference(output_dir, reference_anno_path = None, model_name = 'llama2-7B-chat'):
    if reference_anno_path is None:
        reference_anno_path = os.path.join(output_dir, "plot_contradict_anno.csv")
    anno_df = pd.read_csv(reference_anno_path)
    contradict_df = anno_df[anno_df['type_byAnno'] == 'contradict']
    # For df that is contradict, we sample them all
    pair_list_contradict = list(zip(contradict_df['plot1_id'], contradict_df['plot2_id'], contradict_df['outline_id']))
    not_contradict_df = anno_df[anno_df['type_byAnno'] != 'contradict']
    # For df that is not contradict, we sample 10% of them
    # set random seed
    import random
    random.seed(0)
    sub_not_contradict_df = not_contradict_df.sample(frac=0.1)
    print(f"Number of contradiction: {len(contradict_df)}, Number of not contradiction: {len(not_contradict_df)}")
    print(f"Number of sampled not contradiction: {len(sub_not_contradict_df)}")
    pair_list_notContradict = list(zip(sub_not_contradict_df['plot1_id'], sub_not_contradict_df['plot2_id'], sub_not_contradict_df['outline_id']))
    pair_list = pair_list_contradict + pair_list_notContradict
    curr_pred_dict = inference_given_plot(output_dir, pair_list, model_name= model_name)
    tp, fp, tn, fn = 0, 0, 0, 0
    for k, v in curr_pred_dict.items():
        if v == 'contradict':
            if k in pair_list_contradict:
                tp += 1
            else:
                fp += 1
        else:
            if k in pair_list_contradict:
                fn += 1
            else:
                tn += 1
    print(f"tp: {tp}, fp: {fp*10}, tn: {tn*10}, fn: {fn}")
    precision = tp / (tp + fp*10)
    recall = tp / (tp + fn)
    print(f"f1: {2*precision*recall/(precision+recall)}")
    temp_path = os.path.join(output_dir, "plot_contradict_temp.csv")
    # use the template and save to the output_path
    template_df = pd.read_csv(temp_path)
    # Add the new column "type_byBaseline" with all "unknown"
    template_df['type_byBaseline'] = "not"
    for i in range(len(template_df)):
        plot1_id = template_df.loc[i, 'plot1_id']
        plot2_id = template_df.loc[i, 'plot2_id']
        outline_id = template_df.loc[i, 'outline_id']
        if (plot1_id, plot2_id, outline_id) in curr_pred_dict:
            template_df.loc[i, 'type_byBaseline'] = curr_pred_dict[(plot1_id, plot2_id, outline_id)]
    output_path = os.path.join(output_dir, f"plot_contradict_baseline_{model_name}.csv")
    template_df.to_csv(output_path, index=False)

if __name__ == "__main__":
    result_dir = "/cluster/project/sachan/zhiheng/zhiheng/fact-track/dataPostAnno/1100_1200_pure_simple_detect_llama2-7B"
    # pair_list = [
    #     ("1.3.1", "1.3.2", "1006_pure_simple"),
    #     # ("2.1.3", "3.1.3", "1009_pure_simple"),
    #     # ("2.1.3", "3.3.1", "1009_pure_simple"),
    # ]
    # inference_given_plot(result_dir, pair_list) llama2-7B-chat
    # baseline_inference(result_dir, model_name="gpt-4-1106-preview")
    baseline_inference(result_dir, model_name="llama2-7B-chat")