import pandas as pd
import sys
# This file is mainly to check the relationship between two plot by the existing information of two facts
# 1. get all post-fact of the prev-plot, and the pre-fact of the next-plot
# 2. Check the time relationship of all facts: Overlap, Partial Overlap, Blocked
# 3. Contradict，Not Contradict，Unknown
def calc_PlotPairRelationship(result_dir, metaname, plot_id1, plot_id2):
    # Check if two plot_id is prefix of the each other, if so, return
    if len(plot_id1) > len(plot_id2):
        if plot_id1[:len(plot_id2)] == plot_id2:
            return
    else:
        if plot_id2[:len(plot_id1)] == plot_id1:
            return
    if plot_id1 > plot_id2:
        plot_id1, plot_id2 = plot_id2, plot_id1
    from dataStructure_cleaning import check_fact_pairs
    df_fact = pd.read_csv(result_dir + "fact.csv")
    df_factContradict = pd.read_csv(result_dir + "fact_contradict.csv")
    # print(len(df_factContradict))
    fact_contradict_dict = {}  # Use the dict to accelerate the query
    for idx, row in df_factContradict.iterrows():
        fact_contradict_dict[row["fact1_key"] + "_" + row["fact2_key"]] = row["contradict"]
        if row["contradict"] == True:
            fact_contradict_dict[row["fact1_key"] + "_" + row["fact2_key"]] = "contradict"
        elif row["contradict"] == False:
            fact_contradict_dict[row["fact1_key"] + "_" + row["fact2_key"]] = "not contradict"
    # print(df_factContradict.keys())
    result_dict = {'fact1_key':[], 'fact2_key':[], 'nli_score':[], 'contradict':[], 'fact1_content':[],
       'fact2_content':[]}
    fact_pairs = check_fact_pairs(plot_id1, plot_id2, metaname, df_fact)# plot_idx1, plot_idx2, outline_id, fact_info_df
    sys.path.append("/home/yangk/zhiheng/develop_codeversion/chatGPT-planner")
    from chatGPT_contradictDetector_factDecompose import openai_UnitContradictCheck, huggingface_UnitContradictScore
    from chatGPT_API import load_model2classification
    model = load_model2classification("gpt-4")
    from tqdm import trange
    for i in trange(len(fact_pairs['fact1_id'])):
        fact1_id = fact_pairs['fact1_id'][i]
        fact2_id = fact_pairs['fact2_id'][i]
        if fact1_id + "_" + fact2_id in fact_contradict_dict.keys():
            continue
        fact1 = df_fact[df_fact['fact_key'] == fact1_id].iloc[0].to_dict()
        fact2 = df_fact[df_fact['fact_key'] == fact2_id].iloc[0].to_dict()
        # print(fact1)
        # print(fact2)
        nli_score = huggingface_UnitContradictScore(fact1['fact_content'], fact2['fact_content'])
        if nli_score > 0.05:
            contradict = openai_UnitContradictCheck(fact1['fact_content'], fact2['fact_content'], model)
        else:
            contradict = False
        # print(nli_score)
        # print(contradict)
        result_dict['nli_score'].append(nli_score)
        if contradict:
            result_dict['contradict'].append("contradict afterDetection")
            print("contradict afterDetection")
        else:
            result_dict['contradict'].append("not contradict")
        result_dict['fact1_key'].append(fact1_id)
        result_dict['fact2_key'].append(fact2_id)
        fact1_content = f"{metaname}_{plot_id1}_{fact1['fact_content']}_postfact"
        fact2_content = f"{metaname}_{plot_id2}_{fact2['fact_content']}_prefact"
        result_dict['fact1_content'].append(fact1_content)
        result_dict['fact2_content'].append(fact2_content)
    result_df = pd.DataFrame(result_dict)
    result_df = pd.concat([result_df, df_factContradict], axis=0)
    # print(len(result_df))
    result_df.to_csv(result_dir + "fact_contradict.csv", index=False)
    return result_df

def load_mixDF(result_dir):
    pred_path = result_dir + "plot_contradict_pred.csv"
    anno_path = result_dir + "plot_contradict_anno.csv"
    pred_df = pd.read_csv(pred_path)
    anno_df = pd.read_csv(anno_path)
    if len(pred_df) != len(anno_df):
        raise ValueError("The length of pred_df and anno_df are not equal!")
    # print(pred_df.head())
    # print(anno_df.head())
    # Merge two dataframe
    merge_df = pd.merge(pred_df, anno_df, on=["plot1_id", "plot2_id", "outline_id"], how="inner")
    return merge_df


def plot_crossMatrix(result_dir):
    merge_df = load_mixDF(result_dir)
    print(merge_df.head())
    # Visualize the cross matrix between type_byPred     type_byAnno
    list_pred = merge_df["type_byPred"].tolist()
    list_anno = merge_df["type_byAnno"].tolist()
    import numpy as np
    classes_pred = ['contradict', 'ignored', 'blocked', 'not contradict', 'unknown', 'invalid']
    dict_pred = {}
    for i in range(len(classes_pred)):
        dict_pred[classes_pred[i]] = i
    classes_anno = ['contradict', 'not contradict', 'invalid']
    dict_anno = {}
    for i in range(len(classes_anno)):
        dict_anno[classes_anno[i]] = i
    matrix = np.zeros((len(classes_pred), len(classes_anno)))
    for i in range(len(list_pred)):
        matrix[dict_pred[list_pred[i]], dict_anno[list_anno[i]]] += 1
    print(classes_pred)
    print(classes_anno)
    print(matrix)

def calculate_allImportantPairs(result_dir):
    # return a list of [metaname, plot_id1, plot_id2]
    # the generation is a new ...
    merge_df = load_mixDF(result_dir)
    merge_df = merge_df[merge_df["type_byPred"] == "unknown"]
    merge_df = merge_df[merge_df["type_byAnno"] == "contradict"]
    result_list = []
    for index, row in merge_df.iterrows():
        result_list.append([row["outline_id"], row["plot1_id"], row["plot2_id"]])
    print(result_list)
    return result_list

if __name__ == "__main__":
    result_dir = "/home/yangk/zhiheng/develop_codeversion/fact-track/dataPostAnno/1005_pure_simple/"
    result_list = calculate_allImportantPairs(result_dir)
    print(result_list)
    # exit(0)
    for item in result_list:
        calc_PlotPairRelationship(result_dir, item[0], item[1], item[2])
    # generate_plotContradictAnno(["1001_pure_simple"], result_dir)
    from dataStructure_cleaning import generate_plotContradictPred
    generate_plotContradictPred(["1005_pure_simple"], result_dir)
    plot_crossMatrix(result_dir)