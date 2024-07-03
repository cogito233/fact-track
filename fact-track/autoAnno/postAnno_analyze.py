import pandas as pd
import os
BASE_PATH = os.environ["BASE_PATH"]

def main():
    # Setting: decide which is used as ground truth
    # Step 1: data format conversion, currently is just simply use gpt4 zeroshot
    # Step 2: calculate the f1 value
    # Step 3: analyze the badcase
    result_name = "1100_1200_pure_simple_detect_llama2-7B"
    gt_name = "gpt4_pure_simple_llama2-7B_1100_1200.csv"
    from postAnno_migrate import migrate2plot_contradict_anno
    from postAnno_F1Analyze import f1_analyze
    from postAnno_badcaseAnalyze import print_badcase, save_badcase
    meta_dir = f"{BASE_PATH}/fact-track/dataPostAnno/{result_name}"
    gt_path = f"{BASE_PATH}/fact-track/dataAutoAnno/{gt_name}"
    # print(f"Meta dir: {meta_dir}")
    # print(f"GT path: {gt_path}")
    gt_target_path = meta_dir + "/plot_contradict_anno.csv"
    temp_path = meta_dir + "/plot_contradict_temp.csv"
    if not os.path.exists(gt_target_path):
        migrate2plot_contradict_anno(gt_path, gt_target_path, temp_path)
    pred_path = meta_dir + "/plot_contradict_pred.csv"
    f1_analyze(gt_target_path, pred_path)
    print(f"Pred path: {pred_path}")
    print(f"GT target path: {gt_target_path}")
    print_badcase(meta_dir)
    save_badcase(meta_dir)

if __name__ == "__main__":
    main()