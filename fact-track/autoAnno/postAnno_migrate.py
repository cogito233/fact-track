import pandas as pd
import os
BASE_PATH = os.environ["BASE_PATH"]

import sys
sys.path.append(f"{BASE_PATH}/fact-track/fact-track/analyze")

def clean_idx(idx):
    """
    Function to clean the given index string by collecting the first digit and
    any subsequent '.{int}' parts, then combining them into a clean index.

    :param idx: A string representing the index.
    :return: A cleaned version of the index.
    """
    return idx
    parts = idx.split(".")
    cleaned_parts = [parts[0]]  # Start with the first part which is always a digit

    # Iterate through the parts and add only those that are purely digits
    for part in parts[1:]:
        if part.isdigit():
            cleaned_parts.append(part)

    # Combine the cleaned parts with '.' separator
    clean_idx = '.'.join(cleaned_parts)
    return clean_idx


# Used to merge all content into the standard format
def data_migrate(begin, end, suffix = "_detect_llama2-7B"):
    from dataStructure_cleaning import generate_metaContent, generate_plotContradictPred
    meta_path = f"{BASE_PATH}/fact-track/data"
    result_dir = f"{BASE_PATH}/fact-track/dataPostAnno/{begin}_{end}_pure_simple{suffix}/"
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)
    metaname_list = []
    for i in range(begin, end):
        metaname = f"{i}_pure_simple" # this is a dir
        if os.path.exists(f"{meta_path}/{metaname}{suffix}"):
            metaname_list.append(metaname)
    print(metaname_list)
    generate_metaContent(metaname_list, result_dir, suffix = suffix)
    generate_plotContradictPred(metaname_list, result_dir)
    print(result_dir)

# for a data, we have several types of anno
# GPT4: ZeroShot Anno, RobostAnno_{0..5}
# llama: ZeroShot Anno, RobostAnno_{0..5}
# ...?
# So just leave this function used by other analyze code...
def migrate2plot_contradict_anno(source_path, target_path, temp_path, suffix = "_llama2-7B"):
    # source_path = "/home/yangk/zhiheng/develop_codeversion/fact-track/dataPostAnno/anno_result_1010_1019.csv"
    # # type,idx1,idx2,analyze,reason,is_contradiction,outlineIdx
    # target_path = "/home/yangk/zhiheng/develop_codeversion/fact-track/dataPostAnno/101x_pure_simple/plot_contradict_anno.csv"
    # plot1_id,plot2_id,outline_id,type_byPred
    print(source_path)
    print(target_path)
    print(temp_path)
    temp_df = pd.read_csv(temp_path)
    source_df = pd.read_csv(source_path)
    # temp_df = temp_df[temp_df['outline_id'].isin(source_df['outlineIdx'])]
    target_df = pd.DataFrame(columns=["plot1_id","plot2_id","outline_id","type_byAnno"])
    source_df_dict = {}
    for idx, row in source_df.iterrows():
        if row["is_contradiction"]:
            type_byPred = "contradict"
        else:
            type_byPred = "not contradict"
        # print(row)
        idx1, idx2 = row["idx1"], row["idx2"]
        idx1 = clean_idx(str(idx1))
        idx2 = clean_idx(str(idx2))
        if idx1 > idx2:
            idx1, idx2 = idx2, idx1
        if idx1 == idx2[:len(idx1)]:
            # print("There is a invalid contradiction: ", idx1, idx2)
            pass
        else:
            source_df_dict[(row["outlineIdx"], idx1, idx2)] = type_byPred
            # print(row["outlineIdx"], idx1, idx2, type_byPred)
    print(len(source_df_dict))
    matched = 0
    for idx, row in temp_df.iterrows():
        plot1_id, plot2_id = row["plot1_id"], row["plot2_id"]
        outline_id = row["outline_id"] + suffix
        plot1_id = clean_idx(str(plot1_id))
        plot2_id = clean_idx(str(plot2_id))
        if source_df_dict.get((outline_id, plot1_id, plot2_id),None) is None:
            type_byPred = "not contradict"
        else:
            matched += 1
            type_byPred = source_df_dict[(outline_id, plot1_id, plot2_id)]
        target_df.loc[idx] = [row["plot1_id"],row["plot2_id"],row["outline_id"],type_byPred]
    # print(len(target_df))
    # print(len(temp_df))
    print(f"Matched: {matched}")
    target_df.to_csv(target_path, index=False)
    # exit(0)

if __name__ == "__main__":
    # source_path = "/cluster/project/sachan/zhiheng/zhiheng/fact-track/dataAutoAnno"
    # # type,idx1,idx2,analyze,reason,is_contradiction,outlineIdx
    # target_path = "/home/yangk/zhiheng/develop_codeversion/fact-track/dataPostAnno/101x_pure_simple/anno_result_1010_1019_llama2-7B.csv"
    # source_path = "/cluster/project/sachan/zhiheng/zhiheng/fact-track/dataAutoAnno/gpt3_fullAnno_pure_simple_llama2-7B_1100_1200.csv"
    # target_path = "/cluster/project/sachan/zhiheng/zhiheng/fact-track/dataPostAnno/1100_1200_pure_simple_detect_llama2-7B/plot_contradict_anno_gpt3.csv"
    # temp_path = "/cluster/project/sachan/zhiheng/zhiheng/fact-track/dataPostAnno/1100_1200_pure_simple_detect_llama2-7B/plot_contradict_temp.csv"
    # migrate2plot_contradict_anno(source_path, target_path, temp_path, suffix = "_llama2-7B")
    data_migrate(1100, 1200, suffix = "_llama2-7B_detect_gpt4_d3")

