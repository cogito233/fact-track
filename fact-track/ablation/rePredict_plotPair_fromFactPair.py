import pandas as pd
import numpy as np
import os
import sys
from tqdm import tqdm

def concat_df(path):# first half + last half = full df
    dfname_1 = "fact_contradict_llama_fewshot_first_half.csv" # i<mid
    dfname_2 = "fact_contradict_llama_fewshot_last_half.csv" # i>=mid
    path_1 = path + dfname_1
    path_2 = path + dfname_2
    dfname_out = "fact_contradict_llama_fewshot_full.csv"
    df_1 = pd.read_csv(path_1)
    df_2 = pd.read_csv(path_2)
    mid = len(df_1) // 2
    # select first-half, last_half and concat
    df_1 = df_1.iloc[:mid]
    df_2 = df_2.iloc[mid:]
    df_full = pd.concat([df_1, df_2], ignore_index=True)
    print(len(df_1), len(df_2), len(df_full))
    print(df_full.head())
    df_full.to_csv(path + dfname_out, index=False)

# def reInferencePlotPair(path, fact_contradict_dfName, output_name): # plot_contradict_pred_CR=llama.csv; plot_contradict_pred_CR=gpt4.csv
#     def is_contradict(l1, r1, l2, r2, fact_type1, fact_type2, contradict):
#         if contradict == False:
#             return False
#         if fact_type1 == fact_type2:
#             return False
#         if fact_type1 == "pre_fact":
#             fact_type2, fact_type1 = fact_type1, fact_type2
#             l1, l2 = l2, l1
#             r1, r2 = r2, r1
#         if l2<=l1 and l1<=r2 and r2<=r1:
#             return True
#         else:
#             return False
#     import os
#     fact_df = pd.read_csv(os.path.join(path, "fact.csv"))
#     # plot_id	outline_id	fact_key	fact_content	l	r	fact_type
#     plot_contradict_temp_df = pd.read_csv(os.path.join(path, "plot_contradict_temp.csv"))
#     # outline_id, plot1_id, plot2_id
#     fact_contradict_df = pd.read_csv(os.path.join(path, fact_contradict_dfName))
#     # fact1_key	fact2_key	nli_score	contradict	fact1_content	fact2_content
#
#     # outline_id, plot_id1, plot_id_2 -> contradict_result
#     plot_contradict_temp_dict = {}
#
#     # Generate a dict with all "unknown"
#     for i in range(len(plot_contradict_temp_df)):
#         outline_id = plot_contradict_temp_df.loc[i, "outline_id"]
#         plot_id1 = plot_contradict_temp_df.loc[i, "plot1_id"]
#         plot_id2 = plot_contradict_temp_df.loc[i, "plot2_id"]
#         plot_contradict_temp_dict[(outline_id, plot_id1, plot_id2)] = "unknown"
#
#     from tqdm import trange
#     for i in trange(len(fact_contradict_df)):
#         fact1_key = fact_contradict_df.loc[i, "fact1_key"]
#         fact2_key = fact_contradict_df.loc[i, "fact2_key"]
#         contradict = fact_contradict_df.loc[i, "contradict"]
#         fact1_content = fact_contradict_df.loc[i, "fact1_content"]
#         fact2_content = fact_contradict_df.loc[i, "fact2_content"]
#         fact1 = fact_df[fact_df["fact_key"] == fact1_key].iloc[0]
#         l1, r1, fact_type1 = fact1["l"], fact1["r"], fact1["fact_type"]
#         fact2 = fact_df[fact_df["fact_key"] == fact2_key].iloc[0]
#         l2, r2, fact_type2 = fact2["l"], fact2["r"], fact2["fact_type"]
#         real_contradict = is_contradict(l1, r1, l2, r2, fact_type1, fact_type2, contradict)
#         outline_id = fact1["outline_id"]
#         plot_id1 = fact1["plot_id"]
#         plot_id2 = fact2["plot_id"]
#         if real_contradict:
#             plot_contradict_temp_dict[(outline_id, plot_id1, plot_id2)] = "contradict"
#         # else:
#         #     print("Error: ", outline_id, plot_id1, plot_id2)
#     # save plot_contradict_temp_dict to output_name, with outline_id, plot_id1, plot_id_2, type_byPred
#     output_df = pd.DataFrame(columns=["outline_id", "plot1_id", "plot2_id", "type_byPred"])
#     for key in plot_contradict_temp_dict:
#         outline_id, plot_id1, plot_id2 = key
#         type_byPred = plot_contradict_temp_dict[key]
#         output_df = output_df.append({"outline_id": outline_id, "plot1_id": plot_id1, "plot2_id": plot_id2, "type_byPred": type_byPred}, ignore_index=True)
#     output_df.to_csv(os.path.join(path, output_name), index=False)
#     # Count the number of contradict pairs
#     print(len(output_df[output_df["type_byPred"] == "contradict"]))
#     print(len(output_df[output_df["type_byPred"] == "unknown"]))

def is_contradict(l1, r1, l2, r2, fact_type1, fact_type2, contradict):
    # print(l1, r1, l2, r2, fact_type1, fact_type2, contradict)
    if not contradict or fact_type1 == fact_type2:
        return False
    if fact_type1 == "pre_fact":
        fact_type2, fact_type1 = fact_type1, fact_type2
        l1, l2 = l2, l1
        r1, r2 = r2, r1
    return l2 <= l1 <= r2 and r2 <= r1

def reInferencePlotPair(path, fact_contradict_dfName, output_name):
    fact_df = pd.read_csv(os.path.join(path, "fact.csv"))
    plot_contradict_temp_df = pd.read_csv(os.path.join(path, "plot_contradict_temp.csv"))
    fact_contradict_df = pd.read_csv(os.path.join(path, fact_contradict_dfName))

    fact_lookup = fact_df.set_index('fact_key')
    results = []

    for _, row in tqdm(fact_contradict_df.iterrows(), total=fact_contradict_df.shape[0]):
        fact1 = fact_lookup.loc[row['fact1_key']]
        # If there is multiple fact1_key, select the first one
        if isinstance(fact1, pd.DataFrame):
            fact1 = fact1.iloc[0]
        fact2 = fact_lookup.loc[row['fact2_key']]
        # If there is multiple fact2_key, select the first one
        if isinstance(fact2, pd.DataFrame):
            fact2 = fact2.iloc[0]
        real_contradict = is_contradict(fact1['l'], fact1['r'], fact2['l'], fact2['r'], fact1['fact_type'], fact2['fact_type'], row['contradict'])

        if real_contradict:
            results.append({
                "outline_id": fact1['outline_id'],
                "plot1_id": fact1['plot_id'],
                "plot2_id": fact2['plot_id'],
                "type_byPred": "contradict"
            })

    output_df = pd.DataFrame(results, columns=["outline_id", "plot1_id", "plot2_id", "type_byPred"])
    output_df.to_csv(os.path.join(path, output_name), index=False)
    print(len(output_df[output_df["type_byPred"] == "contradict"]))


def subsample_result(path, input_name, output_name, num=500):
    # Load the dataframe
    df = pd.read_csv(os.path.join(path, input_name))

    # Count the number of contradict pairs
    contradict_count = len(df[df["type_byPred"] == "contradict"])
    print(contradict_count)

    if contradict_count > num:
        # Get indices of all contradict rows
        contradict_indices = df[df["type_byPred"] == "contradict"].index

        # Randomly select num indices to keep as contradict
        keep_indices = np.random.choice(contradict_indices, size=num, replace=False)

        # Set all other contradict rows to unknown
        df.loc[~df.index.isin(keep_indices) & df['type_byPred'].eq('contradict'), 'type_byPred'] = 'unknown'

    # Save the modified DataFrame
    df.to_csv(os.path.join(path, output_name), index=False)
    print(len(df[df["type_byPred"] == "contradict"]))


def exp_result(path):
    import os
    BASE_PATH = os.environ["BASE_PATH"]
    sys.path.append(f"{BASE_PATH}/fact-track/fact-track/oddAnno")
    from scoreAnno_pos import scoreAnnotation
    input_name = "plot_contradict_pred_CR=gpt4_500.csv"
    output_name = "plot_scoreOutline_pred_CR=gpt4_500.csv"
    scoreAnnotation(path, input_name, output_name, is_simple=False)

    input_name = "plot_contradict_pred_CR=llama_500.csv"
    output_name = "plot_scoreOutline_pred_CR=llama_500.csv"
    scoreAnnotation(path, input_name, output_name, is_simple=False)

    from scoreAnno_pos import scoreAnnotation
    input_name = "plot_contradict_pred_CR=gpt4_500.csv"
    output_name = "plot_scoreSimple_pred_CR=gpt4_500.csv"
    scoreAnnotation(path, input_name, output_name, is_simple=False)

    input_name = "plot_contradict_pred_CR=llama_500.csv"
    output_name = "plot_scoreSimple_pred_CR=llama_500.csv"
    scoreAnnotation(path, input_name, output_name, is_simple=False)

def plot_result(path):
    import os
    BASE_PATH = os.environ["BASE_PATH"]
    sys.path.append(f"{BASE_PATH}/fact-track/fact-track/oddAnno")
    from scoreVis_pos import calculate_statistics
    calculate_statistics("plot_scoreOutline_pred_CR=gpt4_500.csv", path)
    calculate_statistics("plot_scoreOutline_pred_CR=llama_500.csv", path)
    calculate_statistics("plot_scoreSimple_pred_CR=gpt4_500.csv", path)
    calculate_statistics("plot_scoreSimple_pred_CR=llama_500.csv", path)

if __name__ == "__main__":
    path = "/home/yangk/zhiheng/fact-track/dataPostAnno/1100_1200_pure_simple_detect_llama2-7B/"
    # concat_df(path)
    # df_name = "/home/yangk/zhiheng/fact-track/dataPostAnno/1100_1200_pure_simple_detect_llama2-7B/fact_contradict_gpt_fewshot_first_half.csv"
    # df = pd.read_csv(df_name)
    # print(len(df))
    # print(df.head())
    # reInferencePlotPair(path, "fact_contradict_llama_fewshot_full.csv", "plot_contradict_pred_CR=llama.csv")
    # # reInferencePlotPair(path, "fact_contradict_gpt_fewshot_first_half.csv", "plot_contradict_pred_CR=gpt4.csv")
    # input_name1 = "plot_contradict_pred_CR=gpt4.csv"
    # input_name2 = "plot_contradict_pred_CR=llama.csv"
    # output_name1 = "plot_contradict_pred_CR=gpt4_500.csv"
    # output_name2 = "plot_contradict_pred_CR=llama_500.csv"
    # subsample_result(path, input_name1, output_name1, num = 500)
    # subsample_result(path, input_name2, output_name2, num = 500)
    exp_result(path)
    plot_result(path)
    #


