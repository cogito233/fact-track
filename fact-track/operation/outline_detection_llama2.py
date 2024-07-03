import sys
sys.path.append("/home/yangk/zhiheng/develop_codeversion/fact-track/fact-track/core")
from outline import OutlineItem, Outline, generate_outline
from contradict_detector import ContradictDetector_StatusPersistence
from state_checker import OutlineItemStateChecker

sys.path.append("/home/yangk/zhiheng/develop_codeversion/fact-track/fact-track/utils")
from llama_api import load_model2generation
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
    # model_temp = 1.0
    model_temp = 0.0
    # Now the main program
    # Initialize all the variables
    model_generation = load_model2generation(temp = model_temp) # Not used here
    model_rewrite = load_model2generation(temp = model_temp) # Not used here
    # model_decomposition = load_model2generation(temp = 0)
    # model_classification = load_model2classification(temp = 0)
    contradictDetector = ContradictDetector_StatusPersistence(model_name_decompose = "llama2",
                                                            model_name_contradict = "huggingface",
                                                            log_file_name = logSaver.metaname + "/contradict_log.log",
                                                            contradict_file_name = logSaver.metaname + "/contradict_list",
                                                            similarity_threshold = 0.2,
                                                            nli_threshold = 0.05,
                                                            same_threshold = 0.99,
                                                            )
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

# Head of outline Injection
sys.path.append("/home/yangk/zhiheng/develop_codeversion/fact-track/fact-track/analyze")
from detection_dataset import outline_analyze

def wrapper_detectionFromOutline(outline, max_depth = 2, output_name = None):
    if output_name is None:
        logSaver = LogSaver("sample_detection")
    else:
        logSaver = LogSaver(output_name)
    outline = generate_outline_withInjection(logSaver, outline = outline, max_depth = max_depth)
    # print(outline)
    logSaver.save()
    return outline

def wrapper_detectionFromGeneration(input_metaname, max_depth = 2, output_name = None):
    path = f"/home/yangk/zhiheng/develop_codeversion/fact-track/data/{input_metaname}/object/outline.pkl"
    import pickle
    with open(path, "rb") as f:
        outline = pickle.load(f)
    wrapper_detectionFromOutline(outline, max_depth = max_depth, output_name = output_name)
    outline_analyze(output_name)

if __name__ == "__main__":
    for i in range(0, 10):
        wrapper_detectionFromGeneration(f"{1010+i}_pure_simple", max_depth = 3, output_name=f"{1010+i}_pure_simple_detect_llama2")