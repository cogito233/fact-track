import os
BASE_PATH = os.environ["BASE_PATH"]

import sys
sys.path.append(f"{BASE_PATH}/fact-track/fact-track/utils")
from log_saver import LogSaver

import pandas as pd
import numpy as np
import os

def load_outline(metaname = None):
    # path = f"/home/yangk/zhiheng/develop_codeversion/fact-track/data/{metaname}/analyze_result"
    # import os
    # if not os.path.exists(path):
    #     os.makedirs(path)
    print(metaname)
    logSaver = LogSaver(metaname)
    export_dict = logSaver.load()
    print(export_dict.keys())
    # exit(0)
    return export_dict


def save_allPair(metaname = "1000_pure_simple_detect"):
    # 参考detection_dataset.py, return a dict about each pair and its label
    export_dict = load_outline(metaname)
    print(export_dict.keys())
    outline = export_dict['outline']
    from convert_dataTemplate import outline2idxs
    idxs = outline2idxs(outline)
    print(idxs)
    pass

def save_allPairAnno(meta_name, pairInfo_dict):
    # return a dict about each pair and its label
    path = f"/home/yangk/zhiheng/develop_codeversion/fact-track/data/{meta_name}/allPairAnno.csv"
    # reschedule the pairInfo_dict and save them to the path, if the content is not valid, then return False
    # {"id1": id1, "id2": id2, "key":f"{id1}_{id2}", "source": "clear" or "might", "type": "fact" or "redundant"}
    # print(meta_name, pairInfo_dict)
    # Save a csv file to the path
    valid_dataPoint = 0
    output_dict = {
        "id1": [],
        "id2": [],
        "key": [],
        "source": [],
        "type": []
    }
    for source in ["clear", "might"]:
        for num in range(1, 16):
            if f"{source}_contradict:::outline_index_1_{num}" not in pairInfo_dict:
                continue
            if f"{source}_contradict:::outline_index_2_{num}" not in pairInfo_dict:
                continue
            if f"{source}_contradict:::contradiction_type_{num}" not in pairInfo_dict:
                continue
            idx1 = pairInfo_dict[f"{source}_contradict:::outline_index_1_{num}"]
            # delete all " " and "," at the end of the string
            if type(idx1) == str:
                idx1 = idx1.strip()
                idx1 = idx1.strip(",")
            idx2 = pairInfo_dict[f"{source}_contradict:::outline_index_2_{num}"]
            if type(idx2) == str:
                idx2 = idx2.strip()
                idx2 = idx2.strip(",")
            type_str = pairInfo_dict[f"{source}_contradict:::contradiction_type_{num}"]
            # if there is any NaN, then break
            if pd.isna(idx1) or pd.isna(idx2) or pd.isna(type):
                continue
            if type_str == "redundant contradiction" or type_str == "factual contradiction":
                # if idx = 1.0 then change it to int
                if type(idx1) != str and int(idx1) == idx1:
                    idx1 = int(idx1)
                if type(idx2) != str and int(idx2) == idx2:
                    idx2 = int(idx2)
                valid_dataPoint += 1
                output_dict["id1"].append(idx1)
                output_dict["id2"].append(idx2)
                output_dict["key"].append(f"{idx1}_{idx2}")
                output_dict["source"].append(source)
                output_dict["type"].append(type_str)
            else:
                return False # Not meet the requirement
            print(idx1, idx2, type_str)
    if valid_dataPoint == 0:
        return False # Return False when the content is not valid or no valid data point
    path_csv = f"/home/yangk/zhiheng/develop_codeversion/fact-track/dataPostAnno/{meta_name}_allPairAnno.csv"
    df = pd.DataFrame(output_dict)
    df.to_csv(path_csv, index=False)
    return True

def mix_allPair(meta_name):

    pass

def parse_csv(path = "/home/yangk/zhiheng/develop_codeversion/fact-track/dataPostAnno/annotation_result_1110.csv"):
    # two functions:
    # 1. parse the annotation result and for each outline, generate two csv files about plot pair relationship
    # 2. re-generate the test for all outlines that haven't been annotated
    # Random comment: maybe I need a script to download the file from the amazon server

    df = pd.read_csv(path, sep='\t')
    # Select the rows that df["instance_id"] end with "pure_simple"
    df_pure_simple = df[df["instance_id"].str.endswith("pure_simple")]
    columns = df_pure_simple.columns
    ids = df_pure_simple["instance_id"]
    print(columns)
    print(ids)
    valid_ids = []
    for i in range(len(ids)):
        id = ids.iloc[i]
        item = df_pure_simple.iloc[i]
        print(item)
        dict_item = item.to_dict()
        # save_allPair(id)
        if save_allPairAnno(id, dict_item):
            valid_ids.append(id)
    print(valid_ids)
    return valid_ids

def check_validity():
    # check the validity of the annotation result, and generate a new_task
    force_reAnnotated = []
    pass

if __name__ == "__main__":
    parse_csv()
    save_allPair()