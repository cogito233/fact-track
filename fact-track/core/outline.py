import json
import random
import os
BASE_PATH = os.environ["BASE_PATH"]

def load_premise():
    path = f"{BASE_PATH}/fact-track/premise/wp.txt"
    with open(path, "r") as f:
        lines = f.readlines()
    return lines

def generate_premise(model):
    prompt = ["Could you generate a premise around 50 words for a long story?\nPremise:"]
    outputs = model(prompt)[0]
    return outputs

def rewrite_premise(premise, model):
    # There are some problems because it may change the meaning of the premise, so we may not use it
    # Even if the premise is bad, we should still use it to write a story faithfully
    # Update: I think that is not my problem, it is the problem of the model, GPT-4 will be better
    prompt = [f""""Premise: {premise}

Could you use one adjective to list the most weakness of the premise above, and use instructions to describe how it can be improved? Please use the follow template.

Adjectives:
Instructions:
    """]
    outputs = model(prompt)[0]
    # print(outputs)
    adj, instruct = outputs.split('\nInstructions: ')
    prompt = [f"""Old Premise: {premise}

Could you rewrite the old premise? {instruct}

New Premise: """]
    outputs = model(prompt)[0]
    return outputs

class OutlineItem:
    def __init__(self, str_outline):
        def normalize(s):
            if s is None:
                return None
            return s.replace('\n', '').replace(':', '').strip()

        # str_outline = str_outline.split('\n')
        print("Now is parsing outline item")
        try:
            main_plot, str_outline = str_outline.split("Characters")
            if "Main plot" in main_plot:
                idx, main_plot = main_plot.split("Main plot")
            else:
                idx, main_plot = main_plot.split(":")
            if "Begin event:" in str_outline:
                characters, str_outline = str_outline.split("Begin event")
                begin_event, str_outline = str_outline.split("End event")
                end_event = str_outline
            else:
                characters = str_outline
                begin_event = None
                end_event = None
            self.idx, self.main_plot, self.characters, self.begin_event, self.end_event = idx, main_plot, characters, begin_event, end_event
        except:
            print(str_outline)
            raise Exception('OutlineItem: str_outline is not valid')
        # idx, main_plot, begin_event, end_event, characters = str_outline[:5]
        # Remove all "\n" and ":", and also space at the beginning and end
        self.idx = normalize(self.idx)
        self.main_plot = normalize(self.main_plot)
        self.begin_event = normalize(self.begin_event)
        self.end_event = normalize(self.end_event)
        self.characters = normalize(self.characters)
        self.has_boundry_event = self.end_event is not None
        print(str(self))

    def __str__(self):
        if self.has_boundry_event:
            return f"Point {self.idx}\nMain plot: {self.main_plot}\nCharacters: {self.characters}\nBegin event: {self.begin_event}\nEnd event: {self.end_event}\n\n"
        else:
            return f"Point {self.idx}\nMain plot: {self.main_plot}\nCharacters: {self.characters}\n\n"

    def str(self, max_depth = None):
        if max_depth == 0:
            return ""
        else:
            return str(self)

    def full_plot(self):
        if self.has_boundry_event:
            return f"{self.main_plot}. At the beginning, {self.begin_event}. At end, {self.end_event}."
        else:
            return f"{self.main_plot}"

    def plot_human(self):
        if self.has_boundry_event:
            return f"""{"-"*(len(str(self.idx))-1)} {self.idx} {self.main_plot}
{"-"*(len(str(self.idx))*2)} Beginning: {self.begin_event}
{"-"*(len(str(self.idx))*2)} End: {self.end_event}"""
        else:
            return f"""{self.idx} {self.main_plot}"""

    def plot_human_noPadding(self):
        if self.has_boundry_event:
            return f"""{self.idx} {self.main_plot}
Beginning: {self.begin_event}
End: {self.end_event}"""
        else:
            return f"""{self.idx} {self.main_plot}"""

    def full_plot_format(self):
        if self.has_boundry_event:
            return f"Main plot: {self.main_plot}\nCharacters: {self.characters}\nBegin event: {self.begin_event}\nEnd event: {self.end_event}"
        else:
            return f"Main plot: {self.main_plot}\nCharacters: {self.characters}"

    def plot_html(self, depth = "0"):
        if self.has_boundry_event:
            return f"""<div class="depth_{depth}"><strong>{self.idx}</strong> {self.main_plot}
    <strong>Beginning:</strong> {self.begin_event}
    <strong>End:</strong> {self.end_event}</div>"""
        else:
            return f"""<div class="depth_{depth}"><strong>{self.idx}</strong> {self.main_plot}</div>"""

    def get_dict(self, max_depth = None):
        if max_depth == 0:
            return {}
        elif self.has_boundry_event:
            return {"idx": self.idx, "main_plot": self.main_plot, "begin_event": self.begin_event, "end_event": self.end_event, "characters": self.characters}
        else:
            return {"idx": self.idx, "main_plot": self.main_plot, "characters": self.characters}

    def get_json(self, max_depth = None):
        return json.dumps(self.get_dict(max_depth = max_depth))

    def expand2outline(self, model, root_outline, prompt_method = "simple", creative_method = False,
                       bandwidth = 3):
        # print("Now is expanding to outline")
        # print("######################################################################")
        idx = self.idx
        # print(root_outline.son_outlines)
        # print(type(root_outline.son_outlines[0]))
        # print(root_outline)
        prompt_outline = root_outline.get_prompt(idx, prompt_method = prompt_method)
        if not creative_method:
            prompt = f"""Can you break down point {idx} into {bandwidth} independent, chronological and similarly-scoped sub-points? Also list the names of characters that appear. Please follow the template below. Include detailed information about each character in the "Main Plot".  Do not answer anything else.

{generate_template(f"{idx}.", bandwidth, root_outline.meta_prompt)}
"""
            # print(prompt_outline + prompt)
            # print(len(prompt_outline + prompt))
            outputs = model([prompt_outline + prompt])[0]
            print("######################################################################")
            print(outputs)
            print("######################################################################")
            events = [event for event in outputs.split('Point ')[1:] if event.count('\n') >= 3]
            # if "\n" in event >= 3
            # print(events)
            new_outline = Outline(events, outline_item=self)
            # print(new_outline)
            # exit(0)
            return new_outline
        else:
            prompt = f"""Can you break down point {idx} into up to 5 potential different storylines with {bandwidth} independent, chronological and similarly-scoped sub-points? Also list the names of characters that appear. Please follow the template below with "Main Plot" and "Characters". Include detailed information about each character in the "Main Plot". Do not answer anything else.

Storyline 1:
{generate_template(f"{idx}.", bandwidth, root_outline.meta_prompt)}

Storyline 2:
{generate_template(f"{idx}.", bandwidth, root_outline.meta_prompt)}

Storyline 3:
{generate_template(f"{idx}.", bandwidth, root_outline.meta_prompt)}

Storyline 4:
{generate_template(f"{idx}.", bandwidth, root_outline.meta_prompt)}

Storyline 5:
{generate_template(f"{idx}.", bandwidth, root_outline.meta_prompt)}
"""
            print(len(prompt_outline + prompt))
            outputs = model([prompt_outline + prompt])[0]

            substory = [event for event in outputs.split('Storyline ')[1:]]
            print(len(substory))
            # Random select a substory and parse it
            substory = random.choice(substory)
            plotPoints = parsing_substory(substory)
            new_outline = Outline(plotPoints, outline_item=self)
            return new_outline

class Outline:
    def __init__(self, son_outlines, premise = None, outline_item = None):
        if premise is not None:
            self.is_root = True
            self.premise = premise
        else:
            self.is_root = False
            self.outline_item = outline_item
        try:
            self.son_outlines = [OutlineItem(outline) for outline in son_outlines]
        except:
            raise Exception('Outline: son_outlines is not valid')
        # print(self.son_outlines)
        # print(son_outlines)
        self.has_boundry_event = self.son_outlines is not None and self.son_outlines[-1].has_boundry_event
        self.meta_prompt = """Main plot: [TODO]
Characters: [TODO]"""
        if self.has_boundry_event:
            self.meta_prompt = """Main plot: [TODO]
Characters: [TODO]
Begin event: [TODO]
End event: [TODO]"""

    def __str__(self):
        if self.is_root:
            result_str = f"Premise: {self.premise}\n\nOutline:\n\n"
            for outline in self.son_outlines:
                result_str += str(outline)
        else:
            result_str = str(self.outline_item)
            for outline in self.son_outlines:
                result_str += str(outline)
        return result_str

    def str(self, max_depth = None):
        if max_depth == 0:
            return ""
        if max_depth is not None:
            max_depth -= 1
        if self.is_root:
            result_str = f"Premise: {self.premise}\n\nOutline:\n\n"
            for outline in self.son_outlines:
                result_str += outline.str(max_depth = max_depth)
        else:
            result_str = str(self.outline_item)
            for outline in self.son_outlines:
                result_str += outline.str(max_depth = max_depth)
        return result_str

    def get_dict(self, max_depth = None):
        if max_depth == 0:
            return None
        if max_depth is not None:
            max_depth -= 1
        if self.is_root:
            result_dict = {"premise": self.premise}
        else:
            result_dict = {"outline_item": self.outline_item.get_dict()}
        if max_depth != 0:
            outline_tree = {}
            for idx in range(len(self.son_outlines)):
                outline = self.son_outlines[idx]
                if type(outline) == OutlineItem:
                    outline_tree[outline.idx] = outline.get_dict()
                else:
                    outline_tree[outline.outline_item.idx] = outline.get_dict_plain(max_depth=max_depth)
            result_dict["outline"] = outline_tree
        return result_dict

    def get_json(self, max_depth = None):
        return json.dumps(self.get_dict(max_depth = max_depth))

    def get_prompt(self, idx, prompt_method = "simple"):
        def check(curr_idx):
            # print(curr_idx, idx)
            if prompt_method == "simple":
                return curr_idx == idx
            elif prompt_method == "detail":
                return len(curr_idx)==1 or curr_idx[:len(curr_idx)-2] == idx[:len(curr_idx)-2]
            elif prompt_method == "full":
                return True
            else:
                raise ValueError("The prompt method is not correct!")
        # simple: only parent
        # detail: ancestors and their siblings
        # full: all
        if self.is_root:
            result_str = f"Premise: {self.premise}\n\nOutline:\n\n"
        else:
            if check(self.outline_item.idx):
                result_str = str(self.outline_item)
            else:
                result_str = ""
        print(self.son_outlines)
        for outline in self.son_outlines:
            if (type(outline) == OutlineItem):
                if check(outline.idx):
                    result_str += str(outline)
            else:
                result_str += outline.get_prompt(idx, prompt_method = prompt_method)
        return result_str

    def get_partial_outline(self, target_idx):
        # Code from dataset_anno/convert_dataTemplate.py
        outlineItem_list = []
        def generate_text(curr_outline):
            # 比较idx的长度，如果一样就比较字典序大小
            current_text = ""
            if type(curr_outline) == Outline:
                if not curr_outline.is_root and len(curr_outline.outline_item.idx) <= len(target_idx):
                    curr_idx = curr_outline.outline_item.idx
                    if len(curr_idx) < len(target_idx) or len(curr_idx) == len(target_idx) and curr_idx < target_idx:
                        current_text += f"{curr_idx} {curr_outline.outline_item.main_plot}\n\n"
                        outlineItem_list.append(curr_outline.outline_item)
                    elif curr_idx == target_idx:
                        current_text += f"{curr_idx} [placeholder]\n\n"
                for son_outline in curr_outline.son_outlines:
                    current_text += generate_text(son_outline)
            elif type(curr_outline) == OutlineItem:
                curr_idx = curr_outline.idx
                if len(curr_idx) < len(target_idx) or len(curr_idx) == len(target_idx) and curr_idx < target_idx:
                    current_text += f"{curr_idx} {curr_outline.main_plot}\n\n"
                    outlineItem_list.append(curr_outline)
                elif curr_idx == target_idx:
                    current_text += f"[placeholder]\n\n"
            else:
                raise ValueError("The type of outline is not correct!")
            return current_text

        outline_text = generate_text(self)
        return outline_text, outlineItem_list

    def rewrite2detail(self, model, root_outline):
        if self.son_outlines is None:
            raise Exception("Incomplete outline")
        for i in range(len(self.son_outlines)):
            outline_item = self.son_outlines[i]
            if type(outline_item) != OutlineItem:
                raise Exception("Incomplete type of son outline")
            idx = outline_item.idx
            # if idx == "1.1":
            #     print("######################################################################")
            #     print(root_outline)
            #     print("######################################################################")
            #     print(root_outline.get_prompt(idx))
            #     print("######################################################################")
            #     exit(0)
            # By default, use simple prompt to rewrite
            prompt_outline = f"""{root_outline.get_prompt(idx)}"""
            prompt = f"""Could you rewrite plot point {idx} in a more detailed way but using only one sentence? To address the vagueness, try adding more sensory details and specific settings to ground the story in a particular place and time. Additionally, consider fleshing out the child character to make them more distinct and memorable. Please use the following template with "Main Plot" and "Characters". Do not answer anything else.

Point {idx}
{self.meta_prompt}
"""
            prompt = prompt_outline + prompt
            # print(len(prompt))
            # print("######################################################################")
            # print(prompt)
            # print("######################################################################")
            count = 0
            while count < 3:
                outputs = model([prompt])[0]
                print(outputs)
                events = [event for event in outputs.split('Point ')[1:]]
                if len(events) == 1:
                    self.son_outlines[i] = OutlineItem(events[0])
                    break
                else:
                    count += 1
                    if count == 3:
                        self.son_outlines[i] = OutlineItem(outputs)
                        # if len(events) == 0:
                        #     raise Exception("No events generated")
                        # else:
                        #     raise Exception("More than one events generated")
            # if len(events) == 0:
            #     raise Exception("No events generated")
            # elif len(events) == 1:
            #     self.son_outlines[i] = OutlineItem(events[0])
            # else:
            #     raise Exception("More than one events generated")

def parsing_substory(substory):
    # print(substory)
    substory = substory.split("Point ")[1:]
    return substory

def generate_outline(premise, model, boundary_event = False, creative_method = False, bandwidth = 3):
    if boundary_event:
        meta_outline = """Main plot: [TODO]
Characters: [TODO]
Begin event: [TODO]
End event: [TODO]"""
    else:
        meta_outline = """Main plot: [TODO]
Characters: [TODO]"""

    if creative_method:
        prompt = f"""Premise:  {premise}

Can you break down the premise into up to 5 potential different storylines with {bandwidth} independent, chronological and similarly-scoped sub-points? Also list the names of characters that appear. Please follow the template below with "Main Plot" and "Characters".  Do not answer anything else.

Storyline 1:
{generate_template("", bandwidth, meta_outline)}

Storyline 2:
{generate_template("", bandwidth, meta_outline)}

Storyline 3:
{generate_template("", bandwidth, meta_outline)}

Storyline 4:
{generate_template("", bandwidth, meta_outline)}

Storyline 5:
{generate_template("", bandwidth, meta_outline)}
"""
        outputs = model([prompt])[0]
        # print("######################################################################")
        # print(model)
        # print(prompt)
        # print("######################################################################")
        # print(model([prompt]))
        # print(outputs)
        # print("######################################################################")
        substory = [event for event in outputs.split('Storyline ')[1:]]
        # print(len(substory))
        # Random select a substory and parse it
        substory = random.choice(substory)
        events = [event for event in substory.split('Point ')[1:]]
        print(events)
        outline = Outline(events, premise=premise)
        # print(outline)
        # exit(0)
    else:
        prompt = f"""Premise: {premise}

Can you break down the premise into {bandwidth} independent, same-scaled plot point? Also, assign each character a name. Please use the following template with "Main Plot" and "Characters". Do not answer anything else.

{generate_template("", bandwidth, meta_outline)}
"""
        outputs = model([prompt])[0]
        events = [event for event in outputs.split('Point ')[1:]]
        outline = Outline(events, premise = premise)
        if outline.son_outlines is None:
            outline = generate_outline(premise, model)
        # recursive generation if failed
    return outline

def generate_template(idx, bandwidth, meta_prompt):
    template = ""
    for i in range(bandwidth):
        template += f"Point {idx}{i+1}\n{meta_prompt}\n\n"
    return template

if __name__ == '__main__':
    # premise = "After years of estrangement, a successful businesswoman receives an unexpected message from her long-lost mother. The message is cryptic and seems to indicate that her mother is in trouble. Despite her initial reluctance, the woman decides to embark on a journey to find her mother and uncover the truth behind the message. Along the way, she discovers long-buried family secrets and comes to terms with the reasons for their estrangement. Will she be able to reconcile with her mother before it's too late?"
    print(generate_template("1.", 3, "Main plot: [TODO]\nCharacters: [TODO]\nBegin event: [TODO]\nEnd event: [TODO]"))