import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "4,5"
BASE_PATH = os.environ["BASE_PATH"]
import sys

sys.path.append(f"{BASE_PATH}/fact-track/fact-track/core")
from outline import OutlineItem, Outline, generate_outline, load_premise

sys.path.append(f"{BASE_PATH}/fact-track/fact-track/utils")
from gpt_api import load_model2classification, load_model2generation
from log_saver import LogSaver

eps = 1e-6

def generate_outline_fromPremise(model, premise, logSaver, max_depth = 2, bandwidth = 3):
    # Need to check the source code?
    # prompt_method = "detail"
    prompt_method = "detail"
    boundary_event = True
    # creative_method = True
    creative_method = False
    rewrite = False
    model_temp = 1.0
    # Now the main program
    maximum_retry = 10
    retry = 0
    while retry < maximum_retry:
        try:
            outline = generate_outline(premise, model, boundary_event = boundary_event,
                                       creative_method = creative_method, bandwidth = bandwidth)
            break
        except:
            retry += 1
            print("Retry: ", retry)
    if retry == maximum_retry:
        raise Exception("Generate outline failed!")
    # print(type(outline.son_outlines[0]))
    # exit(0)
    contradict_detector = None
    if rewrite:
        outline.rewrite2detail(model, outline)
    logSaver.add_outline(outline)
    logSaver.add_model(model_inGeneration = model)
    logSaver.add_detector(contradict_detector)
    # print(outline)
    # Checkpoint 1, check the outline
    # exit(0)
    queue = [[outline, 1, 0, 1]] # [outline, depth, l, r]
    while len(queue) > 0:
        item = queue.pop(0)
        print(item)
        curr_outline, curr_depth = item[0], item[1]
        l, r = item[2] + eps, item[3]
        if curr_depth == max_depth:
            continue
        length = len(curr_outline.son_outlines)
        interval_size = (r - l) / length
        for i in range(length):
            new_l, new_r = l+interval_size*i, l+interval_size*(i+1)-eps
            # print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
            # print("Now is before expand2outline")
            # print(outline.son_outlines)
            # print(type(outline.son_outlines[0]))
            # print(outline)
            maximum_retry = 10
            retry = 0
            while retry < maximum_retry:
                try:
                    new_outline = curr_outline.son_outlines[i].expand2outline(model, outline, prompt_method = prompt_method,
                                                                                creative_method = creative_method, bandwidth = bandwidth)
                    break
                except:
                    retry += 1
                    print("Retry: ", retry)
            if retry == maximum_retry:
                raise Exception("Expand2outline failed!")
            # if type(curr_outline.son_outlines[i]) == Outline:
            #     print("Something wired happened!")
            curr_outline.son_outlines[i] = new_outline
            if rewrite:
                new_outline.rewrite2detail(model, outline)
            queue.append([new_outline, curr_depth + 1, new_l, new_r])
    # print(outline)
    return outline

def wrapper_generation(model, premise, max_depth = 2, bandwidth = 3, output_name = None):
    # TODO: if already exist, then skip
    if output_name is None:
        logSaver = LogSaver()
    else:
        logSaver = LogSaver(output_name)

    # model = load_model2generation(temp = model_temp)
    try:
        outline = generate_outline_fromPremise(model, premise, logSaver, max_depth = max_depth, bandwidth = bandwidth)
    except:
        print("$"*50)
        print(f"Generate outline failed! on {output_name}")
        logSaver.remove()
        return None
    # print(outline)
    logSaver.save()
    return outline

def batch_generation(begin = 1000, end = 1010):
    premise = load_premise()
    print("premise length: ", len(premise))
    import random
    # set seed
    random.seed(0)
    random.shuffle(premise)
    print(premise[0])
    from tqdm import trange
    for i in trange(begin, end):
        model = load_model2generation(temp=model_temp)
        wrapper_generation(model, premise[i], max_depth=3, output_name=str(i)+"_pure_simple")
        # break


def batch_generation_llama2(begin = 1000, end = 1010):
    premise = load_premise()
    print("premise length: ", len(premise))
    import random
    # set seed
    random.seed(0)
    random.shuffle(premise)
    print(premise[0])
    from tqdm import trange
    for i in trange(begin, end):
        from llama_api_vLLM import load_deterministic_llama2
        model = load_deterministic_llama2("7B-chat")
        wrapper_generation(model, premise[i], max_depth=3, output_name=str(i)+"_pure_simple_llama2-7B")

if __name__ == '__main__':
    # premise = "After years of estrangement, a successful businesswoman receives an unexpected message from her long-lost mother. The message is cryptic and seems to indicate that her mother is in trouble. Despite her initial reluctance, the woman decides to embark on a journey to find her mother and uncover the truth behind the message. Along the way, she discovers long-buried family secrets and comes to terms with the reasons for their estrangement. Will she be able to reconcile with her mother before it's too late?"
    # outline = wrapper_generation(premise)
    # print(outline)
    # outline = wrapper_generation(premise, max_depth = 3, bandwidth = 3, output_name="sample_b3d3")
    # outline = wrapper_generation(premise, max_depth = 2, bandwidth = 5, output_name="sample_b5d2_real")
    # batch_generation()#begin = 1002, end = 1003)
    # batch_generation(begin = 999, end = 1000)
    # batch_generation(begin = 1010, end = 1020)
    import torch
    print(torch.__version__)
    print(torch.cuda.is_available())
    print(torch.version.cuda)
    import argparse
    parser = argparse.ArgumentParser(description='batch generation with begin and end number')
    parser.add_argument('--begin', type=int, required=True, help='begin number')
    parser.add_argument('--end', type=int, required=True, help='end number')

    args = parser.parse_args()

    batch_generation_llama2(args.begin, args.end)
