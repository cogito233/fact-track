# Use gpt4-turbo with temp=0.5, and annotate 5 times, visualize the agreement
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import pandas as pd

def anno_robust(input_path, output_dir, temp=0.5, n=5):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    from autoAnno_parsing import load_data, main
    data = load_data(input_path)
    for i in range(n):
        output_path = os.path.join(output_dir, f'anno_{i}.csv')
        if os.path.exists(output_path):
            continue
        main(data, output_path, temp=temp, model = "gpt4") # TODO: need use GPT4

def migrate2plot_contradict_anno_robust(anno_dir):
    def merge_and_calculate(df_list):
        # print(len(df_list[0]))
        # 合并所有DataFrame
        merged_df = pd.concat(df_list)
        # 从合并后的DataFrame中移除'type_byAnno'列
        merged_df = merged_df.drop(columns='type_byAnno')
        # 对原始数据分组，保留'type_byAnno'列用于计算
        grouped = pd.concat(df_list).groupby(['plot1_id', 'plot2_id', 'outline_id'])
        # 计算每个组的num_contradict和type_byAnno
        num_contradict = grouped['type_byAnno'].apply(lambda x: (x == 'contradict').sum())
        type_byAnno = grouped['type_byAnno'].apply(
            lambda x: 'contradict' if 'contradict' in x.values else 'not contradict')
        # 创建一个新的DataFrame来保存这些计算结果
        result_df = pd.DataFrame({'num_contradict': num_contradict, 'type_byAnno': type_byAnno}).reset_index()
        # 将结果与原始DataFrame合并
        final_df = pd.merge(df_list[0], result_df, on=['plot1_id', 'plot2_id', 'outline_id'], how='left')
        # print(len(result_df))
        # exit(0)
        return final_df

    anno_paths = [os.path.join(anno_dir, f'anno_{i}.csv') for i in range(5)]
    target_paths = [os.path.join(anno_dir, f'plot_contradict_anno_{i}.csv') for i in range(5)]
    df_list = []
    for anno_path, target_path in zip(anno_paths, target_paths):
        from postAnno_migrate import migrate2plot_contradict_anno
        # print("anno_path: ", anno_path)
        # print("target_path: ", target_path)
        # exit(0)
        migrate2plot_contradict_anno(anno_path, target_path)
        df_list.append(pd.read_csv(target_path))
        # print(df_list[-1].head())
    # plot1_id,plot2_id,outline_id,type_byAnno, num_contradict
    final_df = merge_and_calculate(df_list)
    output_path = os.path.join(anno_dir, 'plot_contradict_anno.csv')
    final_df.to_csv(output_path, index=False)
    print(len(final_df))

def f1_analyze_withThreshold(anno_dir, pred_path, threshold):
    anno_df = pd.read_csv(os.path.join(anno_dir, 'plot_contradict_anno.csv'))
    pred_df = pd.read_csv(pred_path)
    anno_df['type_byAnno'] = anno_df['num_contradict'].apply(lambda x: 'contradict' if x >= threshold else 'not contradict')
    print(pred_df.head())
    from postAnno_F1Analyze import f1_analyze
    f1_analyze(anno_df, pred_df)

# Step 1: run gpt4-turbo with temp=0.5, and annotate 5 times
# Step 2: migrate it to the given format
# Step 3: plot the F1?
def main_robust(input_path = None, output_dir = None):
    if input_path is None:
        input_path = '/home/yangk/zhiheng/develop_codeversion/fact-track/data2anno/detection_data_1010_1019.csv'
    if output_dir is None:
        output_dir = '/home/yangk/zhiheng/develop_codeversion/fact-track/dataPostAnno/sample_robust_llama2'
    anno_robust(input_path, output_dir) # Annotate 5 times
    migrate2plot_contradict_anno_robust(output_dir) # Migrate to the given format
    pred_df = "/home/yangk/zhiheng/develop_codeversion/fact-track/dataPostAnno/101x_pure_simple/plot_contradict_anno.csv"
    f1_analyze_withThreshold(output_dir, pred_df, threshold=1) # Plot the F1
    anno_df = pd.read_csv(os.path.join(output_dir, 'plot_contradict_anno.csv'))
    f1_analyze(anno_df, pred_df)

if __name__ == '__main__':
    main_robust()