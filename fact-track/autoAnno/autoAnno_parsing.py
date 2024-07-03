import os
BASE_PATH = os.environ["BASE_PATH"]

import sys
sys.path.append(f"{BASE_PATH}/fact-track/fact-track/utils")
print(f"{BASE_PATH}/fact-track/fact-track/utils")
from gpt_api import load_model2generation, determistic_simple_API, non_determistic_simple_API

def load_data(name = "detection_data_1010_1019.csv"):
    path = f"{BASE_PATH}/fact-track/data2anno/{name}"
    import pandas as pd
    data = pd.read_csv(path)
    print(data.head())
    return data


def inference_data(outline, model_name = 'gpt-4-1106-preview', temp = 0):
    # output is a list about contradict pairs
    begin_prompt = """We’re a group of AI researchers aiming to improve models' abilities to detect contradictions in story outline. We will show you a story outline below and ask you to annotate contradictions(including redundancy/repetition of the same events multiple times, or contradictory facts between two events). Please take into account that the outline is structured as a tree. In this tree-like structure, individual points such as 1.1, 1.2, and 1.3 are child nodes of plot point 1, so there is no contradiction between a node with its ancestors such as 1.3 and 1. If the text you enter doesn't match our guidelines, we'll highlight the text box in bold red to alert you.

Please be as comprehensive as possible; many pairs may contradict in only one aspect but are otherwise fine. Here are some examples

Example 1
Event 1: John is taken aback by Linda's words and prepares to respond.
Event 2: John hears Linda and decides to answer.
Label: redundancy contradiction (redundancy)

Example 2
Event 1: John's news shocks Linda, and she's unsure how to react.
Event 2: Linda reacts with confusion and frustration.
Label: factual contradiction (factual: "unsure how to react" vs "reacts")

Example 3
Event 1: John starts responding to Sarah.
Event 2: John tells Sarah he won't comply with her demand.
Label: redundancy contradiction (redundancy)

Example 4
Event 1: Ghosts lead to discovery of Max's evidence. Initially, they leave clues at crime sites. Ultimately, authorities find the same items in Max's home and arrest him.
Event 2: Ghosts disrupt Max's routine. Initially, they alter his routine. Ultimately, his deviations expose him to the authorities.
Label: factual contradiction (factual: "immediate arrest" vs "exposure")

"""
    end_prompt = """Write down the indices of all pairs of events which clearly contradict each other in at least one fact. Use a new line for each pair, and separate using a comma. For example:

factual contradiction | 1.2 | 1.3 | [Analyze: 1.2 mention that communication with Earth already established, but 1.3 indicate it is need be establish] | [Reason: the fact whether communication is established is contradictory] | [is contradiction? (Yes)]
factual contradiction | 2 | 3.3 | [Analyze: 2 mention that linda is angry when she grip an item, 3.3 mention that linda get into angry when shu trun around] | [Reason: based on the temperal order, it seems two diffrent events, independent and not contradict] | [is contradiction? (No)]
factual contradiction | 3.2 | 3.3 | [Analyze: 3.2 ends with the group locating the artifact hidden deep within the school, while 3.3 begins with the group already knowing how to deactivate the artifact, suggesting they have already located it] | [Reason: Since 3.2 happened before 3.3, so it is possible to find before deactivate it.] | [is contradiction? (No)]
etc.

You can just annotate the "highest-level" contradictions-- if e.g., 1.2 contradicts with both 1.3 and its sub-event 1.3.1, you can just write that 1.2 and 1.3 contradict, and omit the contradiction between 1.2 and 1.3.1.

Note:

We anticipate that each annotator will, on average, identify at least 20 pairs of contradictory or might contradictory plot elements in each outline. 
"""
    prompt = f"""{begin_prompt}

{outline}

{end_prompt}"""
    outputs = [non_determistic_simple_API(model_name, prompt, temp = temp)]
    print(outputs[0])
    # exit(0)
    return outputs
    # print(outputs[0])

def inference_data_llama2(outline, use_32K = False):
    import re

    def remove_html_and_replace_click_to_fold(text):
        # Define the regular expression for HTML tags
        html_tag_pattern = re.compile(r'<[^>]+>')
        text = text.replace('<div class="depth_3">', '\n')
        # Replace HTML tags with an empty string
        text_without_html = html_tag_pattern.sub('', text)

        # Replace '[click to fold]' with a newline character
        text_with_replacements = text_without_html.replace('[click to fold]', '\n')

        # Replace '<div class="depth_3">' with a newline character
        return text_with_replacements
    begin_prompt = """We’re a group of AI researchers aiming to improve models' abilities to detect contradictions in story outline. We will show you a story outline below and ask you to annotate contradictions(including redundancy/repetition of the same events multiple times, or contradictory facts between two events). Please take into account that the outline is structured as a tree. In this tree-like structure, individual points such as 1.1, 1.2, and 1.3 are child nodes of plot point 1, so there is no contradiction between a node with its ancestors such as 1.3 and 1. If the text you enter doesn't match our guidelines, we'll highlight the text box in bold red to alert you.

Please be as comprehensive as possible; many pairs may contradict in only one aspect but are otherwise fine. Here are some examples

Example 1
Event 1: John is taken aback by Linda's words and prepares to respond.
Event 2: John hears Linda and decides to answer.
Label: redundancy contradiction (redundancy)

Example 2
Event 1: John's news shocks Linda, and she's unsure how to react.
Event 2: Linda reacts with confusion and frustration.
Label: factual contradiction (factual: "unsure how to react" vs "reacts")

Example 3
Event 1: John starts responding to Sarah.
Event 2: John tells Sarah he won't comply with her demand.
Label: redundancy contradiction (redundancy)

Example 4
Event 1: Ghosts lead to discovery of Max's evidence. Initially, they leave clues at crime sites. Ultimately, authorities find the same items in Max's home and arrest him.
Event 2: Ghosts disrupt Max's routine. Initially, they alter his routine. Ultimately, his deviations expose him to the authorities.
Label: factual contradiction (factual: "immediate arrest" vs "exposure")
"""
    end_prompt = """Write down the indices of all pairs of events which clearly contradict each other in at least one fact. Use a new line for each pair, and separate using a vertical bar. You can just annotate the "highest-level" contradictions-- if e.g., 1.2 contradicts with both 1.3 and its sub-event 1.3.1, you can just write that 1.2 and 1.3 contradict, and omit the contradiction between 1.2 and 1.3.1. And We anticipate that each annotator will, on average, identify at least 20 pairs of contradictory or might contradictory plot elements in each outline. Please only keep the factual contradiction. Please use the following template:

    factual contradiction | 1.2 | 1.3 | [Analyze: 1.2 mention that communication with Earth already established, but 1.3 indicate it is need be establish] | [Reason: the fact whether communication is established is contradictory] | [is contradiction? (Yes)]
    factual contradiction | 2 | 3.3 | [Analyze: 2 mention that linda is angry when she grip an item, 3.3 mention that linda get into angry when shu trun around] | [Reason: based on the temperal order, it seems two diffrent events, independent and not contradict] | [is contradiction? (No)]
    factual contradiction | 3.2 | 3.3 | [Analyze: 3.2 ends with the group locating the artifact hidden deep within the school, while 3.3 begins with the group already knowing how to deactivate the artifact, suggesting they have already located it] | [Reason: Since 3.2 happened before 3.3, so it is possible to find before deactivate it.] | [is contradiction? (No)]
    etc.

"""
    prompt = f"""{begin_prompt}

    {outline}

    {end_prompt}"""
    prompt = remove_html_and_replace_click_to_fold(prompt)
    if use_32K:
        from llama32K_api import load_model_llama
        llama = load_model_llama(temp = 0)
    else:
        from llama_api_vLLM import load_deterministic_llama2
        llama = load_deterministic_llama2("7B-chat")
    print(prompt)
    outputs = llama([prompt])
    print(outputs[0])
    exit(0)
    return outputs

import pandas as pd

def parse_result(text = None):
    if text is None:
        raise ValueError("text is None")

    # Splitting the text into lines and processing each line that starts with "factual contradiction"
    data = []
    for line in text.split("\n"):
        parts = line.split("|")
        if len(parts) == 6:
            # Extracting the relevant parts of the line
            t, idx1, idx2, analyze, reason, is_contradiction = parts
            # Converting the "is_contradiction" part to a boolean
            if "Yes" in is_contradiction or "No" in is_contradiction:
                is_contradiction_bool = True if "Yes" in is_contradiction else False
                # Adding the parsed data to the list
                data.append([t.strip(), idx1.strip(), idx2.strip(), analyze.strip(), reason.strip(), is_contradiction_bool])
        elif len(parts) == 5:
            idx1, idx2, analyze, reason, is_contradiction = parts
            t = "factual contradiction (Not in the template)"
            if "Yes" in is_contradiction or "No" in is_contradiction:
                is_contradiction_bool = True if "Yes" in is_contradiction else False
                # Adding the parsed data to the list
                data.append([t.strip(), idx1.strip(), idx2.strip(), analyze.strip(), reason.strip(), is_contradiction_bool])

    # Creating a DataFrame from the parsed data
    columns = ["type", "idx1", "idx2", "analyze", "reason", "is_contradiction"]
    result_df = pd.DataFrame(data, columns=columns)

    return result_df


def main(data = None, name = "detection_data_1010_1019.csv", temp = 0, model = "gpt4", alias_name = None):
    output_path = f"{BASE_PATH}/fact-track/dataAutoAnno/{model}_{name}"
    if alias_name is not None:
        output_path = f"{BASE_PATH}/fact-track/dataAutoAnno/{model}_{alias_name}_{name}"
    result_df = pd.DataFrame()
    if data is None:
        data = load_data(name)
    from tqdm import trange
    for i in trange(len(data)):
        # if i < 2: # for debugging
        #     continue
        # elif i > 2:
        #     break
        text = data['text'][i]
        if model == "gpt4":
            outputs = inference_data(text, temp = temp)#, model_name="gpt-3.5-turbo-16k")
        elif model == "gpt3":
            outputs = inference_data(text, temp = temp, model_name="gpt-3.5-turbo-16k")
        elif model == "llama2":
            outputs = inference_data_llama2(text)
        elif model == "llama2_32K":
            outputs = inference_data_llama2(text, use_32K = True)
        local_result_df = parse_result(outputs[0])
        print(local_result_df)
        # exit(0)
        local_result_df['outlineIdx'] = data['id'][i]
        result_df = result_df._append(local_result_df)
        result_df.to_csv(output_path, index=False)
        # exit(0)
    result_df.to_csv(output_path, index=False)

def test_parsing():
    output = """factual contradiction | 1.1.1 | 1.2.1 | [Analyze: Both events describe Ninety's actions leading to censorship with the same beginning and end, making them redundant.] | [Reason: The events are identical, thus creating a redundancy contradiction.] | [is contradiction? (Yes)]

factual contradiction | 1.1.1 | 1.3.1 | [Analyze: Both events describe Ninety's actions leading to censorship with the same beginning and end, making them redundant.] | [Reason: The events are identical, thus creating a redundancy contradiction.] | [is contradiction? (Yes)]

factual contradiction | 1.1.2 | 1.2.2 | [Analyze: Both events describe a historian or scholar revealing the truth about Ninety's actions to the public, leading to censorship, with the same beginning and end.] | [Reason: The events are identical, thus creating a redundancy contradiction.] | [is contradiction? (Yes)]

factual contradiction | 1.1.2 | 1.3.2 | [Analyze: Both events describe a historian or scholar revealing the truth about Ninety's actions to the public, leading to censorship, with the same beginning and end.] | [Reason: The events are identical, thus creating a redundancy contradiction.] | [is contradiction? (Yes)]

factual contradiction | 1.1.3 | 1.2.3 | [Analyze: Both events describe the impact of Ninety's actions on his family and community, with similar consequences.] | [Reason: The events are nearly identical, thus creating a redundancy contradiction.] | [is contradiction? (Yes)]

factual contradiction | 1.1.3 | 1.3.3 | [Analyze: Event 1.1.3 discusses the impact on family and loved ones, while 1.3.3 discusses the public's reaction, which could be seen as a broader scope including but not limited to family and loved ones.] | [Reason: The events are not identical but overlap significantly in content, potentially creating a redundancy contradiction.] | [is contradiction? (Yes)]

factual contradiction | 2 | 2.2 | [Analyze: Both events describe Ninety's children seeking therapy or counseling to process their emotions, with the same beginning and end.] | [Reason: The events are identical, thus creating a redundancy contradiction.] | [is contradiction? (Yes)]

factual contradiction | 2 | 2.3 | [Analyze: Both events describe Ninety's children confronting their father's actions directly, with the same beginning and end.] | [Reason: The events are identical, thus creating a redundancy contradiction.] | [is contradiction? (Yes)]

factual contradiction | 2.2 | 2.3 | [Analyze: Both events describe Ninety's children dealing with their father's actions, one through therapy and the other through direct confrontation, but they end with the same resolution of finding closure.] | [Reason: The events are similar and end with the same outcome, which could be seen as a redundancy contradiction.] | [is contradiction? (Yes)]

factual contradiction | 3 | 3.2 | [Analyze: Both events describe a new generation grappling with the consequences of Ninety's actions, with the same beginning and end.] | [Reason: The events are identical, thus creating a redundancy contradiction.] | [is contradiction? (Yes)]

factual contradiction | 3 | 3.3 | [Analyze: Both events describe the main character and their peers working together to create a solution to address the ongoing impact of Ninety's actions, with the same beginning and end.] | [Reason: The events are identical, thus creating a redundancy contradiction.] | [is contradiction? (Yes)]

factual contradiction | 3.2 | 3.3 | [Analyze: Both events describe a group of peers challenging the narrative of Ninety's actions and presenting their findings, with the same beginning and end.] | [Reason: The events are identical, thus creating a redundancy contradiction.] | [is contradiction? (Yes)]
"""
    result_df = parse_result(output)
    print(result_df)

if __name__ == "__main__":
    # python autoAnno_parsing.py --name pure_simple_llama2-7B_1100_1200.csv --model gpt4 --alias_name fullAnno
    # python autoAnno_parsing.py --name pure_simple_llama2-7B_1100_1200.csv --model gpt3 --alias_name fullAnno
    import argparse
    parser = argparse.ArgumentParser(description='Process the arguments for the script.')
    parser.add_argument('--name', type=str, default="detection_data_1010_1019.csv", help='Name of the file')
    parser.add_argument('--model', type=str, default="gpt4", help='Model to use (default: gpt4)')
    parser.add_argument('--alias_name', type=str, default=None, help='Alias name for the output file')

    args = parser.parse_args()
    main(name=args.name, model=args.model, alias_name=args.alias_name)

    # args = parser.parse_args()
    # test_parsing()
    # main(name="pure_simple_llama2-7B_1200_1300.csv")
    # main(name = "detection_data_1010_1019.csv", model = "llama2_32K")
    # main(name="detection_data_1010_1019.csv")