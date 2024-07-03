import sys
sys.path.append("/home/yangk/zhiheng/develop_codeversion/fact-track/fact-track/core")
from outline import OutlineItem, Outline, generate_outline
from contradict_detector import ContradictDetector_StatusPersistence
from state_checker import OutlineItemStateChecker

sys.path.append("/home/yangk/zhiheng/develop_codeversion/fact-track/fact-track/utils")
from gpt_api import load_model2classification, load_model2generation
from log_saver import LogSaver

eps = 1e-6

def generate_outline_withInjection(logSaver, premise = None, outline = None, max_depth = 2, injection = None, keep_both = True, bandwidth = 3):
    # injection: ["fact", "plot", None]
    # keep_both means use both injection method and save them
    # If there is no premise, it means we can not modify the current outline? make the injection into None;
    # Although if keep_both = True, we can still try to use some method to correction
    # if premise == None and injection != None:
    #     raise Exception("If there is no premise, then the injection method should be None!")
    if premise == None and outline == None:
        raise Exception("If there is no premise, there should be pre-existing outline!")
    prompt_method = "detail"
    boundary_event = True
    creative_method = True
    rewrite = False
    use_fullPlot = True
    model_temp = 1.0
    # Now the main program
    # Initialize all the variables
    model_generation = load_model2generation(temp = model_temp)
    model_rewrite = load_model2generation(temp = model_temp)
    # model_decomposition = load_model2generation(temp = 0)
    # model_classification = load_model2classification(temp = 0)
    contradictDetector = ContradictDetector_StatusPersistence(model_name_decompose="gpt-4",
                                                              log_file_name = logSaver.metaname + "/contradict_log.log",
                                                              contradict_file_name = logSaver.metaname + "/contradict_list")
    model_decomposition = contradictDetector.model_decompose
    model_classification = contradictDetector.model_contradict

    if premise != None: # That means we need to generate the outline, otherwise we only need to do the check step
        outline = generate_outline(premise, model_generation, boundary_event = boundary_event, creative_method = creative_method)
        if rewrite:
            outline.rewrite2detail(model_generation, outline)

    logSaver.add_outline(outline)
    logSaver.add_model(model_inGeneration = model_generation, model_inRewrite = model_rewrite,
                       model_inDecomposition = model_decomposition, model_inDetection = model_classification)
    logSaver.add_detector(contradictDetector)

    # TODO: Copy from generator, need to change it to detection
    queue = [[outline, 1, 0, 1]]  # [outline, depth, l, r]
    while len(queue) > 0:
        # Step 0: get the current outlineItem from the queue
        item = queue.pop(0)
        print(item) # each item is a slibing of outlineItem
        curr_outline, curr_depth = item[0], item[1]
        l, r = item[2] + eps, item[3]

        length = len(curr_outline.son_outlines)
        interval_size = (r - l) / length
        for i in range(length):
            new_l, new_r = l + interval_size * i, l + interval_size * (i + 1) - eps
            curr_soneOutline = curr_outline.son_outlines[i]
            curr_outlineItem = curr_soneOutline if type(curr_soneOutline) == OutlineItem else curr_soneOutline.outline_item
            print("Current OutlineItem: ", curr_outlineItem)
            print("Current Interval: ", new_l, new_r)
            print("#"*50)
            # Step 1: check whether current outlineItem has problem, if so, fix it, if can not fix, then continue the loop
            curr_stateChecker = OutlineItemStateChecker(curr_outlineItem, new_l, new_r, contradictDetector, outline,
                                                        model_rewrite = model_rewrite, use_fullPlot = use_fullPlot)
            logSaver.add_stateChecker(curr_outlineItem.idx, curr_stateChecker)
            curr_stateChecker.fact_decompose()
            if curr_stateChecker.fact_check():  # It means there is a error occur
                print("#"*100)
                print("There is a error occur!")
                print(curr_stateChecker.observation_dict)
                print("#"*100)
                if injection != None:
                    new_stateChecker = curr_stateChecker.fact_inject(method=injection, keep_both=keep_both)
                    if type(curr_soneOutline) == OutlineItem:
                        curr_outline.son_outlines[i] = new_stateChecker.outlineItem
                    else:
                        curr_outline.son_outlines[i].outline_item = new_stateChecker.outlineItem
                    if new_stateChecker == None:
                        # Failed to fix the problem
                        # Maybe need update curr_stateChecker?
                        continue
                    logSaver.add_stateChecker(curr_outlineItem.idx + "_new", new_stateChecker)
                    new_stateChecker.fact_update()
                else:
                    curr_stateChecker.fact_update()
                    # Update the world status without correction, only work for more fair evaluation of the detection method
            else:
                curr_stateChecker.fact_update()
            # Step 2: expand the outlineItem if there is no problem, if the son is pre-exist, then do not expand it
            if curr_depth == max_depth:
                continue
            if type(curr_soneOutline) == OutlineItem:
                new_outline = curr_outline.son_outlines[i].expand2outline(model_generation, outline,
                                                                          prompt_method=prompt_method,
                                                                          creative_method=creative_method,
                                                                          bandwidth=bandwidth)
                curr_outline.son_outlines[i] = new_outline
                if rewrite:
                    new_outline.rewrite2detail(model_generation, outline)
            if type(curr_outline.son_outlines[i]) == OutlineItem:
                continue
            queue.append([curr_outline.son_outlines[i], curr_depth + 1, new_l, new_r])
    # print(outline)
    logSaver.save()
    return outline

def wrapper_generationFromPremise_withInjection(premise, max_depth = 2, output_name = None, bandwidth = 3):
    if output_name is None:
        logSaver = LogSaver("sample_generation")
    else:
        logSaver = LogSaver(output_name)
    outline = generate_outline_withInjection(logSaver, premise = premise, max_depth = max_depth, injection = "plot_fact", bandwidth = bandwidth)
    # print(outline)
    logSaver.save()
    return outline

def wrapper_detectionFromOutline(outline, max_depth = 2, output_name = None):
    if output_name is None:
        logSaver = LogSaver("sample_detection")
    else:
        logSaver = LogSaver(output_name)
    outline = generate_outline_withInjection(logSaver, outline = outline, max_depth = max_depth)
    # print(outline)
    logSaver.save()
    return outline


def batch_injection(begin = 1000, end = 1010):
    def load_premise():
        path = "/home/yangk/zhiheng/develop_codeversion/chatGPT-planner/analyze_code/dataset/premise_dataset.txt"
        with open(path, "r") as f:
            lines = f.readlines()
        return lines
    premise = load_premise()
    print("premise length: ", len(premise))
    import random
    # set seed
    random.seed(0)
    random.shuffle(premise)
    print(premise[0])
    from tqdm import trange
    for i in trange(begin, end):
        wrapper_generationFromPremise_withInjection(premise[i], max_depth=3, output_name=str(i)+"_injection")
        # break

if __name__ == "__main__":
    # Test the detection function
    # sys.path.append("/home/yangk/zhiheng/develop_codeversion/fact-track/fact-track/test")
    # from outline_stubs import load_outline
    #
    # outline = load_outline()
    # print(outline)
    # outline = wrapper_detectionFromOutline(outline)
    # print(outline)
    # exit(0)
    # Test the generation function
    premise = "After years of estrangement, a successful businesswoman receives an unexpected message from her long-lost mother. The message is cryptic and seems to indicate that her mother is in trouble. Despite her initial reluctance, the woman decides to embark on a journey to find her mother and uncover the truth behind the message. Along the way, she discovers long-buried family secrets and comes to terms with the reasons for their estrangement. Will she be able to reconcile with her mother before it's too late?"
    wrapper_generationFromPremise_withInjection(premise, max_depth = 3, output_name = "sample_generation_b3d3", bandwidth=3)
    wrapper_generationFromPremise_withInjection(premise, max_depth = 2, output_name = "sample_generation_b5d2", bandwidth=5)
    # print(outline)
    # batch_injection(1002, 1004)