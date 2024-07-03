import sys
from outline_stubs import load_outline_stub

sys.path.append("/home/yangk/zhiheng/develop_codeversion/fact-track/fact-track/core")
from outline import OutlineItem, Outline, generate_outline
from contradict_detector import ContradictDetector_StatusPersistence
from state_checker import OutlineItemStateChecker

sys.path.append("/home/yangk/zhiheng/develop_codeversion/fact-track/fact-track/utils")
from gpt_api import load_model2classification, load_model2generation
from log_saver import LogSaver

def test_stateChecker():
    outline = load_outline_stub()
    contradictDetector = ContradictDetector_StatusPersistence(model_name_decompose="gpt-4")
    model_rewrite = load_model2generation(temp = 1.0)
    premise = outline.premise

    # check the pre fact and static fact.
    # For plot1, the l and r is 0 and 0.999999
    new_l, new_r = 0, 0.999999
    curr_outlineItem = outline.son_outlines[0]
    curr_stateChecker = OutlineItemStateChecker(curr_outlineItem, new_l, new_r, contradictDetector, outline,
                                                model_rewrite=model_rewrite, use_fullPlot = True)
    curr_stateChecker.fact_decompose()
    print(curr_stateChecker.fact_check())
    curr_stateChecker.fact_update()

    print("#" * 100)

    # For plot2, the l and r is 1 and 1.999999
    new_l, new_r = 1, 1.999999
    curr_outlineItem = outline.son_outlines[1]
    curr_stateChecker = OutlineItemStateChecker(curr_outlineItem, new_l, new_r, contradictDetector, outline,
                                                model_rewrite=model_rewrite, use_fullPlot = True)
    curr_stateChecker.fact_decompose()
    if curr_stateChecker.fact_check():  # It means there is a error occur
        print("#" * 100)
        print("There is a error occur!")
        print(curr_stateChecker.observation_dict)
        print("#" * 100)
        if premise != None:
            new_stateChecker = curr_stateChecker.fact_inject(method="fact", keep_both=True)
            if new_stateChecker == None:
                # Failed to fix the problem
                # Maybe need update curr_stateChecker?
                print("Failed to fix the problem!")
            # logSaver.add_stateChecker(curr_outlineItem.idx + "_new", curr_stateChecker)
            new_stateChecker.fact_update()
            # Update the world status without
    else:
        curr_stateChecker.fact_update()

    contradictDetector.end_log()

if __name__ == "__main__":
    test_stateChecker()