import pandas as pd
# Same code at analyze/postAnno_badcaseAnalyze.py:print_badcase
def f1_analyze(anno_path, pred_path):
    if type(anno_path) == str:
        anno_df = pd.read_csv(anno_path)
    elif type(anno_path) == pd.DataFrame:
        anno_df = anno_path
    if type(pred_path) == str:
        pred_df = pd.read_csv(pred_path)
    elif type(pred_path) == pd.DataFrame:
        pred_df = pred_path
    anno_df = anno_df[anno_df['outline_id'].isin(pred_df['outline_id'])]
    fp, fn = 0, 0
    tp, tn = 0, 0
    from tqdm import trange
    key_pred = 'type_byPred' if 'type_byPred' in pred_df.columns else 'type_byAnno'
    key_anno = 'type_byAnno' if 'type_byAnno' in anno_df.columns else 'type_byPred'
    print(pred_df.head())
    print(anno_df.head())
    print(len(pred_df), len(anno_df))
    for i in trange(len(pred_df)):
        label = pred_df.iloc[i][key_pred]
        gt_label = anno_df.iloc[i][key_anno]
        if gt_label == "contradict":
            if label == "contradict":
                tp += 1
            else:
                fn += 1
        else:
            if label == "contradict":
                fp += 1
            elif label != "not sampled":
                tn += 1
    print(f"tp: {tp}, tn: {tn}")
    print(f"fp: {fp}, fn: {fn}")
    f1 = 2*tp/(2*tp+fp+fn)
    print(f"f1: {f1}")


if __name__ == "__main__":
    # pred_path = "/home/yangk/zhiheng/develop_codeversion/fact-track/dataPostAnno/101x_pure_simple_llama2-7B_block=0.8/plot_contradict_pred.csv"3
    # anno_path = "/home/yangk/zhiheng/develop_codeversion/fact-track/dataPostAnno/101x_pure_simple/plot_contradict_anno.csv"
    # f1_analyze(anno_path, pred_path)

    pred_path = "/home/yangk/zhiheng/develop_codeversion/fact-track/dataPostAnno/101x_pure_simple_llama2-7B_block=0.5/plot_contradict_pred.csv"
    anno_path = "/home/yangk/zhiheng/develop_codeversion/fact-track/dataPostAnno/101x_pure_simple/plot_contradict_anno.csv"
    f1_analyze(anno_path, pred_path)