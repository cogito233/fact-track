import os
import pandas as pd
import numpy as np
from scoreAnno_badcase import inference_score_simple, inference_score_outline

def scoreAnnotation(path, input_name, output_name, is_simple=True):
    if is_simple:
        simple_column = "score_simple"
    else:
        simple_column = "score_outline"
    pred_path = os.path.join(path, input_name)
    plot_path = os.path.join(path, "plot.csv")
    pred_df = pd.read_csv(pred_path)
    plot_df = pd.read_csv(plot_path)
    plot_dict = {(row['plot_id'], row['outline_id']): row['plot_content'] for index, row in plot_df.iterrows()}
    # Determine the prediction column name.
    pred_column = "type_byPred" if "type_byPred" in pred_df.columns else (
        "type_byAnno" if "type_byAnno" in pred_df.columns else "type_byBaseline")
    print("pred_column: ", pred_column)
    print(pred_df[pred_column].value_counts())

    if os.path.exists(os.path.join(path, output_name)):
        # Show the distribution of the score column.
        pred_path = os.path.join(path, output_name)
        pred_df = pd.read_csv(pred_path)
        print(pred_df[simple_column].value_counts())
        print("Output file already exists. Skipping.")
        return
    # Initialize the score_simple column.
    pred_df[simple_column] = None
    # Iterate through each row to compute score_simple, with error handling.
    # Find indices where `pred_column` equals "contradict"
    contradict_indices = pred_df[pred_df[pred_column] == "contradict"].index
    # Randomly sample 500 of these indices (if there are more than 500)
    if len(contradict_indices) > 500:
        sampled_indices = np.random.choice(contradict_indices, size=500, replace=False)
    else:
        sampled_indices = contradict_indices
    # Convert sampled_indices to a set for faster membership testing
    sampled_indices_set = set(sampled_indices)
    from tqdm import tqdm
    for index, row in tqdm(pred_df.iterrows()):
        if index not in sampled_indices_set:
            continue
        try:
            plot1_id = row['plot1_id']
            plot2_id = row['plot2_id']
            outline_id = row['outline_id']
            plot1 = plot_dict[(plot1_id, outline_id)]
            plot2 = plot_dict[(plot2_id, outline_id)]
            # print("plot1: ", plot1)
            # print("plot2: ", plot2)
            if is_simple:
                pred_df.at[index, simple_column] = inference_score_simple(plot1, plot2)
            else:
                pred_df.at[index, simple_column] = inference_score_outline(outline_id, plot1, plot2)
        except Exception as e:
            # Handle any errors by setting the score to "Format invalid".
            pred_df.at[index, simple_column] = "Format invalid"
            print("Error at index {}: {}".format(index, e))
    # Save the modified dataframe to an output file.
    output_path = os.path.join(path, output_name)
    pred_df.to_csv(output_path, index=False)
    # Optional: Visualize the distribution of the pred column.
    print(pred_df[simple_column].value_counts())

def maximum_subsample(is_simple = True):
    path = "/cluster/project/sachan/zhiheng/zhiheng/fact-track/dataPostAnno/1100_1200_pure_simple_detect_llama2-7B/badcase_gpt4Anno&llamaDect.csv"
    df = pd.read_csv(path)
    if is_simple:
        output_name = "plot_scoreSimple_maxNLI.csv"
    else:
        output_name = "plot_scoreOutline_maxNLI.csv"
    # 选择max_nli值最大的500个样本
    df_top500 = df.nlargest(500, 'max_nli')
    simple_column = "score_simple" if is_simple else "score_outline"
    # 检查输出文件是否存在
    output_path = os.path.join(os.path.dirname(path), output_name)
    if os.path.exists(output_path):
        # 显示得分列的分布
        pred_df = pd.read_csv(output_path)
        print(pred_df[simple_column].value_counts())
        print("Output file already exists. Skipping.")
        return
    # 初始化得分列
    df_top500[simple_column] = None
    # 对每个样本计算得分
    from tqdm import tqdm
    for index, row in tqdm(df_top500.iterrows()):
        try:
            plot1 = row['plot_1']
            plot2 = row['plot_2']
            outline_id = row['outline_id']
            # 根据is_simple参数决定使用哪种得分函数
            if is_simple:
                df_top500.at[index, simple_column] = inference_score_simple(plot1, plot2)
            else:
                df_top500.at[index, simple_column] = inference_score_outline(outline_id, plot1, plot2)
        except Exception as e:
            # 发生错误时设置得分为"Format invalid"
            df_top500.at[index, simple_column] = "Format invalid"
            print(f"Error at index {index}: {e}")
    # 保存修改后的DataFrame到输出文件
    df_top500.to_csv(output_path, index=False)
    # 可选：可视化得分列的分布
    print(df_top500[simple_column].value_counts())

def maximum_subsample(is_simple = True):
    path = "/cluster/project/sachan/zhiheng/zhiheng/fact-track/dataPostAnno/1100_1200_pure_simple_detect_llama2-7B/badcase_gpt4Anno&llamaDect.csv"
    df = pd.read_csv(path)
    if is_simple:
        output_name = "plot_scoreSimple_maxNLI.csv"
    else:
        output_name = "plot_scoreOutline_maxNLI.csv"
    # 选择max_nli值最大的500个样本
    df_top500 = df.nlargest(500, 'max_nli')
    simple_column = "score_simple" if is_simple else "score_outline"
    # 检查输出文件是否存在
    output_path = os.path.join(os.path.dirname(path), output_name)
    if os.path.exists(output_path):
        # 显示得分列的分布
        pred_df = pd.read_csv(output_path)
        print(pred_df[simple_column].value_counts())
        print("Output file already exists. Skipping.")
        return
    # 初始化得分列
    df_top500[simple_column] = None
    # 对每个样本计算得分
    from tqdm import tqdm
    for index, row in tqdm(df_top500.iterrows()):
        try:
            plot1 = row['plot_1']
            plot2 = row['plot_2']
            outline_id = row['outline_id']
            # 根据is_simple参数决定使用哪种得分函数
            if is_simple:
                df_top500.at[index, simple_column] = inference_score_simple(plot1, plot2)
            else:
                df_top500.at[index, simple_column] = inference_score_outline(outline_id, plot1, plot2)
        except Exception as e:
            # 发生错误时设置得分为"Format invalid"
            df_top500.at[index, simple_column] = "Format invalid"
            print(f"Error at index {index}: {e}")
    # 保存修改后的DataFrame到输出文件
    df_top500.to_csv(output_path, index=False)
    # 可选：可视化得分列的分布
    print(df_top500[simple_column].value_counts())


def maximum_subsample_300(is_simple = True):
    path = "/home/yangk/zhiheng/fact-track/dataPostAnno/1100_1200_pure_simple_llama2-7B_detect_gpt4_d3/1.csv"
    # Not work, maybe rewrite a function to calaulate the score and subsample
    df = pd.read_csv(path)
    if is_simple:
        output_name = "plot_scoreSimple_maxNLI_300.csv"
    else:
        output_name = "plot_scoreOutline_maxNLI_300.csv"
    # 选择max_nli值最大的500个样本
    df_top500 = df.nlargest(300, 'max_nli')
    simple_column = "score_simple" if is_simple else "score_outline"
    # 检查输出文件是否存在
    output_path = os.path.join(os.path.dirname(path), output_name)
    if os.path.exists(output_path):
        # 显示得分列的分布
        pred_df = pd.read_csv(output_path)
        print(pred_df[simple_column].value_counts())
        print("Output file already exists. Skipping.")
        return
    # 初始化得分列
    df_top500[simple_column] = None
    # 对每个样本计算得分
    from tqdm import tqdm
    for index, row in tqdm(df_top500.iterrows()):
        try:
            plot1 = row['plot_1']
            plot2 = row['plot_2']
            outline_id = row['outline_id']
            # 根据is_simple参数决定使用哪种得分函数
            if is_simple:
                df_top500.at[index, simple_column] = inference_score_simple(plot1, plot2)
            else:
                df_top500.at[index, simple_column] = inference_score_outline(outline_id, plot1, plot2)
        except Exception as e:
            # 发生错误时设置得分为"Format invalid"
            df_top500.at[index, simple_column] = "Format invalid"
            print(f"Error at index {index}: {e}")
    # 保存修改后的DataFrame到输出文件
    df_top500.to_csv(output_path, index=False)
    # 可选：可视化得分列的分布
    print(df_top500[simple_column].value_counts())

def simple_exp():
    path = "/home/yangk/zhiheng/fact-track/dataPostAnno/1100_1200_pure_simple_detect_llama2-7B"
    # input_name = "plot_contradict_anno.csv"
    # output_name = "plot_scoreSimple_anno.csv"
    # scoreAnnotation(path, input_name, output_name)

    # input_name = "plot_contradict_baseline_llama2-7B-chat.csv"
    # output_name = "plot_scoreSimple_baselineLlama.csv"
    # scoreAnnotation(path, input_name, output_name)

    # input_name = "plot_contradict_pred.csv"
    # output_name = "plot_scoreSimple_pred.csv"
    # scoreAnnotation(path, input_name, output_name)

    # input_name = "plot_contradict_random.csv"
    # output_name = "plot_scoreSimple_random.csv"
    # scoreAnnotation(path, input_name, output_name)

    # input_name = "plot_contradict_anno_full.csv"
    # output_name = "plot_scoreSimple_anno_full.csv"
    # scoreAnnotation(path, input_name, output_name)

    # input_name = "plot_contradict_anno_gpt3.csv"
    # output_name = "plot_scoreSimple_anno_gpt3.csv"
    # scoreAnnotation(path, input_name, output_name)
    # for i in range(4):
    #     input_name = f"plot_contradict_overlap{i}.csv"
    #     output_name = f"plot_scoreSimple_overlap{i}.csv"
    #     scoreAnnotation(path, input_name, output_name)


    input_name = "plot_contradict_maxNLI=300_gpt4.csv"
    output_name = "plot_scoreSimple_maxNLI=300_gpt4.csv"
    scoreAnnotation(path, input_name, output_name, is_simple=True)

    # maximum_subsample_300()

def outline_exp():
    path = "/home/yangk/zhiheng/fact-track/dataPostAnno/1100_1200_pure_simple_detect_llama2-7B"
    # input_name = "plot_contradict_anno.csv"
    # output_name = "plot_scoreOutline_anno.csv"
    # scoreAnnotation(path, input_name, output_name, is_simple=False)

    # input_name = "plot_contradict_baseline_llama2-7B-chat.csv"
    # output_name = "plot_scoreOutline_baselineLlama.csv"
    # scoreAnnotation(path, input_name, output_name, is_simple=False)

    # input_name = "plot_contradict_pred.csv"
    # output_name = "plot_scoreOutline_pred.csv"
    # scoreAnnotation(path, input_name, output_name, is_simple=False)

    # input_name = "plot_contradict_random.csv"
    # output_name = "plot_scoreOutline_random.csv"
    # scoreAnnotation(path, input_name, output_name, is_simple=False)

    # input_name = "plot_contradict_anno_full.csv"
    # output_name = "plot_scoreOutline_anno_full.csv"
    # scoreAnnotation(path, input_name, output_name, is_simple=False)

    # input_name = "plot_contradict_anno_gpt3.csv"
    # output_name = "plot_scoreOutline_anno_gpt3.csv"
    # scoreAnnotation(path, input_name, output_name, is_simple=False)

    # for i in range(3, 4):
    #     input_name = f"plot_contradict_overlap{i}.csv"
    #     output_name = f"plot_scoreOutline_overlap{i}.csv"
    #     scoreAnnotation(path, input_name, output_name, is_simple = False)

    input_name = "plot_contradict_maxNLI=300_gpt4.csv"
    output_name = "plot_scoreOutline_maxNLI=300_gpt4.csv"
    scoreAnnotation(path, input_name, output_name, is_simple=False)

if __name__ == "__main__":
    simple_exp()
    # outline_exp()
    # maximum_subsample_300()
    # maximum_subsample_300(False)



