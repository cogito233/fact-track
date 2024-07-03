import os
import pickle
import numpy as np
BASE_PATH = os.environ["BASE_PATH"]

import sys
sys.path.append(f"{BASE_PATH}/fact-track/fact-track/core")

from outline import Outline, OutlineItem

def outline2text(outline):
    # Generate the text we want from a outline
    outline_text = ""
    if type(outline) == Outline:
        if outline.is_root:
            # result_str += f"Premise: {outline.premise}\n\n"
            pass
        else:
            outlineItem = outline.outline_item
            outline_text += f"{outlineItem.idx} {outlineItem.main_plot}\n\n"
        for son_outline in outline.son_outlines:
            outline_text += outline2text(son_outline)
    elif type(outline) == OutlineItem:
        outline_text += f"{outline.idx} {outline.main_plot}\n\n"
    else:
        raise ValueError("The type of outline is not correct!")
    return outline_text

def outline2text_full(outline, depth = 0):
    # Generate the text we want from a outline
    outline_text = ""
    if type(outline) == Outline:
        if outline.is_root:
            # result_str += f"Premise: {outline.premise}\n\n"
            outline_text += '<div>'
        else:
            outlineItem = outline.outline_item
            outline_text += f"{outlineItem.plot_html(depth = depth)}"
            outline_text += '<div class="toggle">[click to fold]</div><div class="content" style="display: block;">'
        for son_outline in outline.son_outlines:
            outline_text += outline2text_full(son_outline, depth = depth+1)
        outline_text += '</div>'
    elif type(outline) == OutlineItem:
        outline_text += f"{outline.plot_html(depth = depth)}"
    else:
        raise ValueError("The type of outline is not correct!")
    return outline_text

def outline2idxs(outline):
    # return a list of idxs by BFS order
    idx_list = []
    queue = [outline]
    while len(queue) > 0:
        outline = queue.pop(0)
        if type(outline) == Outline:
            if outline.is_root:
                pass
            else:
                outlineItem = outline.outline_item
                idx_list.append(outlineItem.idx)
            for son_outline in outline.son_outlines:
                queue.append(son_outline)
        elif type(outline) == OutlineItem:
            idx_list.append(outline.idx)
    return idx_list

def outline_idx2text(outline):
    # return a dict of idxs to text
    idx2text = {}
    queue = [outline]
    while len(queue) > 0:
        outline = queue.pop(0)
        if type(outline) == Outline:
            if outline.is_root:
                pass
            else:
                outlineItem = outline.outline_item
                idx2text[outlineItem.idx] = f"{outlineItem.idx}: {outlineItem.full_plot()}"
            for son_outline in outline.son_outlines:
                queue.append(son_outline)
        elif type(outline) == OutlineItem:
            idx2text[outline.idx] = f"{outline.idx}: {outline.full_plot()}"
    return idx2text

# We have the final outline, then we need to generate the template
def outline2textInGeneration(outline, target_idx):
    outlineItem_list = []
    def generate_text(curr_outline):
        # 比较idx的长度，如果一样就比较字典序大小
        current_text = ""
        if type(curr_outline) == Outline:
            if not curr_outline.is_root and len(curr_outline.outline_item.idx) <= len(target_idx):
                curr_idx = curr_outline.outline_item.idx
                if len(curr_idx) < len(target_idx) or len(curr_idx) == len(target_idx) and curr_idx < target_idx:
                    current_text += f"{curr_outline.outline_item.plot_human()}\n\n"
                    outlineItem_list.append(curr_outline.outline_item)
                elif curr_idx == target_idx:
                    current_text += f"{curr_idx} [placeholder]\n\n"
            for son_outline in curr_outline.son_outlines:
                current_text += generate_text(son_outline)
        elif type(curr_outline) == OutlineItem:
            curr_idx = curr_outline.idx
            if len(curr_idx) < len(target_idx) or len(curr_idx) == len(target_idx) and curr_idx < target_idx:
                current_text += f"{curr_outline.plot_human()}\n\n"
                outlineItem_list.append(curr_outline)
            elif curr_idx == target_idx:
                current_text += f"{curr_idx} [placeholder]\n\n"
        else:
            raise ValueError("The type of outline is not correct!")
        return current_text
    outline_text = generate_text(outline)
    return outline_text, outlineItem_list

def metaNames2detectionData(metanames, output_name = "detection_data.csv"):
    path = f"{BASE_PATH}/fact-track/data2anno"
    result_df = {"outline_idx": [], "premise": [], "outline_text": [], "outline_text_full": [], "text": []}
    import pandas as pd
    for i in metanames:
        from detection_dataset import outline_analyze
        outline_analyze(i)
        local_path = (f"{BASE_PATH}/fact-track/data/{i}"
                      f"/analyze_result/outline_annotation.csv")
        df = pd.read_csv(local_path)
        for idx, row in df.iterrows():
            result_df["outline_idx"].append(row["outline_idx"])
            result_df["premise"].append(row["premise"])
            result_df["outline_text"].append(row["outline_text"])
            result_df["outline_text_full"].append(row["outline_text_full"])
            text = f"""<strong>Premise</strong><br>
{row["premise"]}
<strong>Outline</strong><br>
{row["outline_text_full"]}
"""
            result_df['text'].append(text)
    result_df['id'] = result_df['outline_idx']
    # result_df['text'] = df['outline_text']
    df = pd.DataFrame(result_df)
    print(len(df))
    print(df.head())
    print(f"{path}/{output_name}")
    df.to_csv(f"{path}/{output_name}")

def metaNames2injectionData(metanames, output_name = "injection_data.csv", column_A ="choise_plot", column_B = "choise_fact"):
    # This code is abolished
    raise ValueError("This code is abolished")
    def parse_text(text):
        # print(text)
        list_plot = text.split("\n")
        list_afterfilter = []
        for i in list_plot:
            if len(i)>3:
                list_afterfilter.append(i)
        main_plot, character, begin, end = list_afterfilter[0], list_afterfilter[1], list_afterfilter[2], list_afterfilter[0]
        main_plot = main_plot[len("Main plot: "):]
        character = character[len("Characters: "):]
        begin = begin[len("Begin event: "):]
        end = end[len("End event: "):]
        return f"""{main_plot}
--- Begin: {begin}
--- End: {end}"""
    
    path = "/home/yangk/zhiheng/develop_codeversion/fact-track/data2anno"
    result_df = {"template 1": [], "control 1": [], "is_coumn_A 1": [],
                 "template 2": [], "control 2": [], "is_coumn_A 2": [],
                 "template 3": [], "control 3": [], "is_coumn_A 3": [],
                 "template 4": [], "control 4": [], "is_coumn_A 4": [],
                 "template 5": [], "control 5": [], "is_coumn_A 5": [],
                 "metaname": []
                 }
    import pandas as pd
    for i in metanames:
        local_path = (f"/home/yangk/zhiheng/develop_codeversion/fact-track/data/" +
                      f"{i}/analyze_result/choise_compare.csv")
        df = pd.read_csv(local_path)
        for j in range(5):
            if len(df) <= j:
                template = ""
                control = 0
                is_column_A = False
            else:
                template = df.iloc[j]["template"]
                text_A, text_B = df.iloc[j][column_A], df.iloc[j][column_B]
                text_A, text_B = parse_text(text_A), parse_text(text_B)
                # Randomly choose one of the two options
                is_column_A = False # np.random.randint(0, 2)
                if is_column_A:
                    text_A, text_B = text_B, text_A
                template_placeholders = f"""[Choose between the two options below, in terms of factual consistency with the rest of the outline.]
(A) {text_A}
(B) {text_B}"""
                # replace the [placeholder] with the template_placeholders
                template = template.replace("[placeholder]", template_placeholders)
                control = 1
                is_column_A = True
            result_df[f"template {j+1}"].append(template)
            result_df[f"control {j+1}"].append(control)
            result_df[f"is_coumn_A {j+1}"].append(is_column_A)
        result_df["metaname"].append(i)
    df = pd.DataFrame(result_df)
    df.to_csv(f"{path}/{output_name}")


def metaNames2injectionData_pure(metanames, output_name="injection_data_pure.csv", column_A="choise_plot",
                                 column_B= "choise_fact"):
    raise ValueError("This code is not used now")
    def parse_text(text):
        # print(text)
        list_plot = text.split("\n")
        list_afterfilter = []
        for i in list_plot:
            if len(i) > 3:
                list_afterfilter.append(i)
        main_plot, character, begin, end = list_afterfilter[0], list_afterfilter[1], list_afterfilter[2], \
        list_afterfilter[0]
        main_plot = main_plot[len("Main plot: "):]
        character = character[len("Characters: "):]
        begin = begin[len("Begin event: "):]
        end = end[len("End event: "):]
        return f"""{main_plot}
--- Begin: {begin}
--- End: {end}"""

    path = "/home/yangk/zhiheng/develop_codeversion/fact-track/data2anno"
    result_df = {"template": [], "is_coumn_A": [],
                 "metaname": [], "id": [], "text": []
                 }
    import pandas as pd
    for i in metanames:
        local_path = (f"/home/yangk/zhiheng/develop_codeversion/fact-track/data/" +
                      f"{i}/analyze_result/choise_compare.csv")
        df = pd.read_csv(local_path)
        for j in range(100):
            # iterate for all j
            if len(df) <= j:
                template = ""
                control = 0
                is_column_A = False
            else:
                template = df.iloc[j]["template"]
                text_A, text_B = df.iloc[j][column_A], df.iloc[j][column_B]
                text_A, text_B = parse_text(text_A), parse_text(text_B)
                # Randomly choose one of the two options
                is_column_A = False  # np.random.randint(0, 2)
                if is_column_A:
                    text_A, text_B = text_B, text_A
                template_placeholders = f"""[Choose between the two options below, in terms of factual consistency with the rest of the outline.]
(A) {text_A}
(B) {text_B}"""
                # replace the [placeholder] with the template_placeholders
                template = template.replace("[placeholder]", template_placeholders)
                control = 1
                is_column_A = True
            result_df[f"template"].append(template)
            result_df[f"is_coumn_A"].append(is_column_A)
            result_df["metaname"].append(i)
            result_df["id"].append(f"{i}_{j}")

    # result_df['id'] = result_df['metaname']
    result_df['text'] = result_df['template']
    df = pd.DataFrame(result_df)
    df.to_csv(f"{path}/{output_name}")

if __name__ == "__main__":
    # outline_path = "/home/yangk/zhiheng/develop_codeversion/fact-track/data/sample/object/outline.pkl"
    # outline = pickle.load(open(outline_path, "rb"))
    # # # print(outline_idx2text(outline))
    # print(outline)
    # print(outline2text(outline))
    # print(outline2text_full(outline))
    # # #print(outline2idxs(outline))
    # exit(0)
    # outline_text, outlineItem_list = outline2textInGeneration(outline, "1.2")
    # print(outline_text)
    #print(outlineItem_list)

    metanames = [f"{i}_pure_simple" for i in range(1010, 1019)]
    metaNames2detectionData(metanames, output_name = "detection_data_1010_1019.csv")
    # metanames = ['sample_generation_b3d3', 'sample_generation_b5d2']
    # metanames = ['test_injection']
    # metaNames2injectionData_pure(metanames)


