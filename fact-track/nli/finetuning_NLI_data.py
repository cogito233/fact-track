import pandas as pd
candidate_path = [
    "/home/yangk/zhiheng/develop_codeversion/chatGPT-planner/analyze_code/detection/Outline_44_depth3_factContradict.csv",
    "/home/yangk/zhiheng/develop_codeversion/chatGPT-planner/analyze_code/corruption/MPE_44_new_factContradict.csv",
    "/home/yangk/zhiheng/develop_codeversion/chatGPT-planner/contradictDetector_log/contradict_GPT4.csv",
    "/home/yangk/zhiheng/develop_codeversion/chatGPT-planner/contradictDetector_log/gpt-4_contradict_factPair_dataInWild.csv"
]

def visualize_distribution(path):
    df = pd.read_csv(path)
    # print(df.head())
    length = len(df)
    label_list = []
    premise_list = []
    hypothesis_list = []
    for i in range(length):
        label = df["label"][i]
        if label == -1:
            continue
        if label == 1:
            label = True
        elif label == 0:
            label = False
        label_list.append(label)
        premise_list.append(df["fact1"][i])
        hypothesis_list.append(df["fact2"][i])
    # the length of label_list and the num of True
    print("the length of label_list: ", len(label_list))
    print("the num of True: ", label_list.count(True))
    print(df.head())
    # exit(0)
    pass

# Output API: text and label, save to huggingface dataset
def load_dataset(path):
    df = pd.read_csv(path)
    import datasets
    df = df[df['label'] != -1]
    df['label'] = df['label'].apply(lambda x: True if x == 1 else False)
    df['labels'] = df['label']
    # if df have column plot_idx, then show plot_idx = 72
    if 'plot_idx' in df.columns:
        #sub_df = df[df['plot_idx'] == 73]
        fact1 = df['fact1'].tolist()
        fact2 = df['fact2'].tolist()
        for i in range(len(fact1)):
            if "Harper" in fact1[i] or "Harper" in fact2[i]:
                print(df.iloc[i])
    dataset = datasets.Dataset.from_pandas(df)

    # print(dataset)
    return dataset
    pass

if __name__ == "__main__":
    path = "/home/yangk/zhiheng/develop_codeversion/chatGPT-planner/data_human_label/csv_dataset/contradict_similarity_subset_gpt4.csv"
    df = pd.read_csv(path)
    #print(df.head())
    # print keys
    # print(df.keys())
    df = df[df['skip'] != 1]
    #df = df[df['pred_gpt3.5'] == 1]
    print(df[df['outline_idx']==73])
    exit(0)
    visualize_distribution(candidate_path[0]) # On depth-3 outline; n = 466, pos = 17
    visualize_distribution(candidate_path[1]) # On 10 corrupted Pair; n = 69, pos = 5
    visualize_distribution(candidate_path[2]) # On 5 pairs; n = 151, pos = 6, def test_allpair() in dfs detect
    visualize_distribution(candidate_path[3]) # On 53 pairs; n = 271, pos = 91, inferenceContradictPairInWild() in dfs detect
    train_dataset = load_dataset(candidate_path[3])
    test_dataset = load_dataset(candidate_path[0])
    print(train_dataset)
    print(test_dataset)