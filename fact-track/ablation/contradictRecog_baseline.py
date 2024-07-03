import pandas as pd
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "5,7"
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
BASE_PATH = os.environ["BASE_PATH"]
import sys
sys.path.append(f"{BASE_PATH}/fact-track/fact-track/utils")

def unitContradictCheck(fact1, fact2, model):
    prompt = f"Do the following statements contradict each other? Answer \"Yes\" or \"No\".\n\n{fact1}\n{fact2}\n"
    answer = model([prompt])[0]
    answer = model([prompt])[0].split("\n")
    while len(answer[-1]) == 0 or "please" in answer[-1] or "Please" in answer[-1]:
        if len(answer) == 0:
            return False
        answer = answer[:-1]
    answer = answer[-1]
    print(answer)
    return True if ("Yes" in answer) or ("YES" in answer) else False


def unitContradictCheck_parallel(fact1s, fact2s, model):
    prompts = [f"Do the following statements contradict each other? Answer \"Yes\" or \"No\".\n\n{fact1}\n{fact2}\n" for fact1, fact2 in zip(fact1s, fact2s)]
    # prompt = "Do the following statements contradict each other? Answer \"Yes\" or \"No\".\n\n{fact1}\n{fact2}\n"
    results = model(prompts)
    answers = []
    for result in results:
        answer = result.split("\n")
        flag = True
        while len(answer[-1]) == 0 or "please" in answer[-1] or "Please" in answer[-1]:
            if len(answer) == 0:
                answers.append(False)
                flag = False
                break
            answer = answer[:-1]
        answer = answer[-1]
        if flag:
            answers.append(True if ("Yes" in answer) or ("YES" in answer) else False)
    return answers

def fewshotContradictCheck(fact1, fact2, model):
    try:
        prompt = f"""Do the following statements contradict each other? Only Answer \"Yes\" or \"No\".
Fact 1: John's lifestyle is strictly aligned with the teachings of his faith.
Fact 2: John holds certain religious beliefs before his encounter with the entities.
Answer: No

Fact 1: The society in Europe was functioning normally without any widespread fear or despair.
Fact 2: The populace of Europe is living in fear and despair due to the Black Death.
Answer: Yes

Fact 1: Emily was living a normal life without any chaos or fear related to supernatural experiences.
Fact 2: The demon inside Emily had a certain level of control over her.
Answer: Yes

Fact 1: The selection process has started.
Fact 2: The selection process continues to progress.
Answer: No

Fact 1: The footage contains information that can be analyzed.	
Fact 2: John has access to the footage from the camera.
Answer: No

Fact 1: The townsfolk are healthy and not infected with the mysterious virus.
Fact 2: The infection is causing the townsfolk to behave strangely.
Answer: Yes

Fact 1: {fact1}
Fact 2: {fact2}
    Answer: """
        answer = model([prompt])[0].split("\n")
        while len(answer[-1]) == 0 or (not ("Yes" in answer[-1]) and not ("YES" in answer[-1]) and not ("No" in answer[-1]) and not ("NO" in answer[-1])):
            if len(answer) == 0:
                return False
            answer = answer[:-1]
        answer = answer[-1]
        print(answer)
        return True if ("Yes" in answer) or ("YES" in answer) else False
    except:
        return False

def load_gpt(): # Load gpt4-turbo
    from gpt_api import load_model2classification
    model = load_model2classification("gpt-4-turbo")
    return model

def load_llama(): # Load llama2-7B-chat
    from llama_api_vLLM import load_deterministic_llama2
    model = load_deterministic_llama2("7B-chat")
    return model
    pass

if __name__ == "__main__":
    fact1 = "The townsfolk are healthy and not infected with the virus."
    fact2 = "Alex is viewed as a God by the infected townspeople."
    model = load_llama()
    fact1s = [fact1, fact1, fact1, fact1, fact1]
    fact2s = [fact2, fact2, fact2, fact2, fact2]
    print(unitContradictCheck_parallel(fact1s, fact2s, model))
    # print(unitContradictCheck(fact1, fact2, model))
    # print(fewshotContradictCheck(fact1, fact2, model))
    # print("--------------------------------------------------")
    # model = load_gpt()
    # print(unitContradictCheck(fact1, fact2, model))
    # print(fewshotContradictCheck(fact1, fact2, model))

