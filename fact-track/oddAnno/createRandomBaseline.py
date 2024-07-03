import pandas as pd

import os
BASE_PATH = os.environ["BASE_PATH"]

def vis_distribution(dir, name = None):
    if name == None:
        name = "plot_contradict_anno.csv"
    df = pd.read_csv(f"{dir}/{name}")
    print(df.head())
    # if exist column "type_byPred" then count the distribution of that
    # if not, then count the distribution of "type_byAnno"
    if "type_byPred" in df.columns:
        print(len(df[df['type_byPred'] == "contradict"]["outline_id"].value_counts().keys()))
        print(df["type_byPred"].value_counts())
        # return the number of "contradict"
        return df["type_byPred"].value_counts()["contradict"]
    else:
        print(len(df[df['type_byAnno'] == "contradict"]["outline_id"].value_counts().keys()))
        print(df["type_byAnno"].value_counts())
        # return the number of "contradict"
        return df["type_byAnno"].value_counts()["contradict"]

def generate_randomBaseline(dir, name = None, output_name = None):
    if name == None:
        name = "plot_contradict_anno.csv"
    if output_name == None:
        output_name = "plot_contradict_random.csv"
    num_contradict = vis_distribution(dir, name)
    df = pd.read_csv(f"{dir}/plot_contradict_temp.csv")
    # Random put "contradict" in the column type_byPred in temp df
    # Step 1. generate a list with all non-contradict
    non_contradict_list = ["not contradict"] * (len(df)-num_contradict) + ["contradict"] * num_contradict
    # Step 2. random shuffle the list and put it into the temp df
    import random
    # seed = 0
    random.seed(0)
    random.shuffle(non_contradict_list)
    df["type_byPred"] = non_contradict_list
    print(df.head())
    df.to_csv(f"{dir}/{output_name}", index=False)

if __name__ == "__main__":
    # vis_distribution(f"{BASE_PATH}/fact-track/dataPostAnno/1100_1200_pure_simple_detect_llama2-7B", "plot_contradict_pred.csv")
    generate_randomBaseline(f"{BASE_PATH}/fact-track/dataPostAnno/1100_1200_pure_simple_detect_llama2-7B")
    vis_distribution(f"{BASE_PATH}/fact-track/dataPostAnno/1100_1200_pure_simple_detect_llama2-7B",
                    "plot_contradict_random.csv")