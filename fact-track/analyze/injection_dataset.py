import sys
sys.path.append("/home/yangk/zhiheng/develop_codeversion/fact-track/fact-track/utils")
from log_saver import LogSaver

sys.path.append("/home/yangk/zhiheng/develop_codeversion/fact-track/fact-track/utils")
from gpt_api import load_model2classification, load_model2generation

import sys
sys.path.append("/home/yangk/zhiheng/develop_codeversion/fact-track/fact-track/core")
from outline import OutlineItem, Outline, generate_outline
from contradict_detector import ContradictDetector_StatusPersistence
from state_checker import OutlineItemStateChecker

def inject_template(template, choise_dict):
    choises = ""
    for i in choise_dict:
        choises += f"\nChoise {i}\n {choise_dict[i]}\n"
    # replace [placeholder] in template with choises
    template = template.replace("[placeholder]", choises)
    return template

def outline_analyze(metaname = "sample"):
    def convert_format(plain_outlineItem):
        # TODO: if needed, change the format into more easy to read version
        pass
    path = f"/home/yangk/zhiheng/develop_codeversion/fact-track/data/{metaname}/analyze_result"
    import os
    if not os.path.exists(path):
        os.makedirs(path)
    logSaver = LogSaver(metaname)
    export_dict = logSaver.load()

    from detection_dataset import outline_analyze as outline_analyze_detection
    # export_detection_dataset(metaname, export_dict['outline'], path)
    outline_analyze_detection(metaname)

    print(export_dict['stateChecker_dict'].keys())
    result_df = {"template":[], "choise_plain":[], "choise_fact":[], "choise_plot":[], "choise_mix":[], "choise_gt":[]}
    outline = export_dict['outline']

    for i in export_dict['stateChecker_dict']:
        # if end with "_new", then analyze, otherwise, skip
        if i[-4:] != "_new":
            continue
        i_origin = i[:-4]
        stateChecker = export_dict['stateChecker_dict'][i_origin]
        stateChecker.model_rewrite = load_model2generation(temp = 1.0)
        from convert_dataTemplate import outline2textInGeneration
        template, _ = outline2textInGeneration(outline, i_origin)
        choise_plain = stateChecker.outlineItem.full_plot_format()
        choise_plot = stateChecker.rewrite_byPlotInject()
        choise_fact = stateChecker.rewrite_byFactInject()
        choise_mix = stateChecker.rewrite_byPlotFactInject()
        choise_gt = export_dict['stateChecker_dict'][i].outlineItem.full_plot_format()
        result_df["template"].append(template)
        result_df["choise_plain"].append(choise_plain)
        result_df["choise_fact"].append(choise_fact)
        result_df["choise_plot"].append(choise_plot)
        result_df["choise_mix"].append(choise_mix)
        result_df["choise_gt"].append(choise_gt)
    import pandas as pd
    df = pd.DataFrame(result_df)
    df.to_csv(f"{path}/choise_compare.csv", index=False)
    # write to choise_compare.txt by inject_template
    with open(f"{path}/choise_compare.txt", "w") as f:
        for i in range(len(result_df["template"])):
            template = result_df["template"][i]
            choise_dict = {"plain":result_df["choise_plain"][i], "fact":result_df["choise_fact"][i], "plot":result_df["choise_plot"][i]}
            f.write(inject_template(template, choise_dict)+"\n\n")

def convert_format(metaname_list):
    pass

if __name__ == "__main__":
    # outline_analyze("test_injection")
    outline_analyze("sample_generation_b5d2")
    outline_analyze("sample_generation_b3d3")