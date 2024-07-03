import os
BASE_PATH = os.environ["BASE_PATH"]

import sys
sys.path.append(f"{BASE_PATH}/fact-track/fact-track/utils")
from gpt_api import load_model2classification, load_model2generation
from huggingface_api import get_embedding_contriever, similarity_from_embedding, huggingface_UnitContradictScore

from outline import OutlineItem, Outline, generate_outline
from contradict_detector import ContradictDetector_StatusPersistence

eps = 1e-6

class OutlineItemStateChecker(object):
    def __init__(self, outlineItem, l, r, contradictDetector, root_outline, model_rewrite = None, use_fullPlot = False):
        self.outlineItem = outlineItem
        self.l = l
        self.r = r
        self.contradictDetector = contradictDetector
        self.root_outline = root_outline
        self.preFact = None
        # {"fact": fact, "interval": [l_fact, r_fact]}
        self.postFact = None
        self.staticFact = None
        if model_rewrite == None:
            self.model_rewrite = load_model2generation(temp = 1.0)
        else:
            self.model_rewrite = model_rewrite
            # we need use the same rewrite model for while outline generation
        self.use_fullPlot = use_fullPlot
        self.observation_dict = None
        # For Reference:
        # {"idx", "plot", "error",
        #  "facts":[
        #  {"error fact",
        #  "error fact interval",
        #  "exist fact":[{"fact", "fact interval", "plot", "plot id"}]
        #  ]}]
        #  }
        self.retrieveCache_similarity = None
        self.retrieveCache_contradict = None
        self.rewriteCache_similarity = None
        self.rewriteCache_contradict = None
        self.rewriteCache_mix = None


    def fact_decompose(self):
        plot_text, l, r, idx, contradictDetector = self.outlineItem.main_plot, self.l, self.r, self.outlineItem.idx, self.contradictDetector
        if self.use_fullPlot:
            plot_text = self.outlineItem.full_plot()
        print("Now is fact decompose!!!!!!!!!")
        print(plot_text)
        print("^"*100)
        # Code from injection/outline_generation_BFS.py
        prefact, postfact, staticfact = contradictDetector.fact_decompose(plot_text)
        # Find the l, r of each fact
        prefact_list, postfact_list, staticfact_list = [], [], []
        for fact in prefact:
            l_fact, r_fact = contradictDetector.interval_blockCheck(fact, l, idx, isPreFact=True)
            prefact_list.append({"fact": fact, "interval": [l_fact, r_fact]})
        for fact in postfact:
            l_fact, r_fact = contradictDetector.interval_blockCheck(fact, r, idx, isPreFact=False)
            postfact_list.append({"fact": fact, "interval": [l_fact, r_fact]})
        for fact in staticfact:
            l_fact, r_fact = contradictDetector.interval_blockCheck(fact, l, idx, isPreFact=True)
            staticfact_list.append({"fact": fact, "interval_l": [l_fact, r_fact]})
            l_fact, r_fact = contradictDetector.interval_blockCheck(fact, r, idx, isPreFact=False)
            staticfact_list[-1]["interval_r"] = [l_fact, r_fact]
        self.preFact, self.postFact, self.staticFact = prefact_list, postfact_list, staticfact_list

    def fact_check(self):
        plot_text, l, r, idx, contradictDetector = self.outlineItem.main_plot, self.l, self.r, self.outlineItem.idx, self.contradictDetector
        if self.use_fullPlot:
            plot_text = self.outlineItem.full_plot()
        # Code from injection/outline_generation_BFS.py

        # Check if the fact is contradict, and save to observation_result
        def run_fact_check(fact, l, r, idx, isPrefact):
            isContradict, meta_info = contradictDetector.interval_contradictCheck(fact, l, r, idx, isPreFact=isPrefact)
            # {"fact": fact_inWorld[0], "fact interval": [fact_inWorld[2], fact_inWorld[3]],
            #  "plot": fact_inWorld[-1], "plot id": fact_inWorld[-2], "isPreFact": isPreFact,
            #  "nli_score": fact_inWorld[1]}

            # print(meta_info)
            if isContradict:
                print("Contradict is detected!")
                print(meta_info)
                print(f"Fact: {fact}, Idx: {idx}, Plot: {plot_text}")
            # for fact_info in meta_info:
            #     fact_info['plot'] = plot_text
            #     fact_info['plot'] = plot_GTdict[fact_info['plot id']]
            # If isContradict is False, then meta_info is None
            flag_local = False
            if isContradict:
                flag_local = True
                observation_dict["facts"].append({
                    "error fact": fact,
                    "error fact interval": [l, r],
                    "exist fact": meta_info})
            return flag_local

        if self.preFact == None:
            raise Exception("Please run fact_decompose first!")
        prefact_list, postfact_list, staticfact_list = self.preFact, self.postFact, self.staticFact
        flag = False
        observation_dict = { # Local status
            "idx": idx,
            "plot": plot_text,
            "error": flag,
            "facts": []
        }
        for fact_dict in prefact_list:
            fact, l, r = fact_dict["fact"], fact_dict["interval"][0], fact_dict["interval"][1]
            flag = run_fact_check(fact, l, r, idx, isPrefact=True) or flag
        for fact_dict in postfact_list:
            fact, l, r = fact_dict["fact"], fact_dict["interval"][0], fact_dict["interval"][1]
            flag = run_fact_check(fact, l, r, idx, isPrefact=False) or flag
        for fact_dict in staticfact_list:
            fact, l, r = fact_dict["fact"], fact_dict["interval_l"][0], fact_dict["interval_l"][1]
            flag = run_fact_check(fact, l, r, idx, isPrefact=True) or flag
            l, r = fact_dict["interval_r"][0], fact_dict["interval_r"][1]
            flag = run_fact_check(fact, l, r, idx, isPrefact=False) or flag
        observation_dict["error"] = flag
        # observation_result.append(observation_dict)
        # observation_result_dict[plot_text] = observation_dict
        self.observation_dict = observation_dict
        return flag

    def fact_update(self):
        plot_text, l, r, idx, contradictDetector = self.outlineItem.main_plot, self.l, self.r, self.outlineItem.idx, self.contradictDetector
        if self.use_fullPlot:
            plot_text = self.outlineItem.full_plot()
        # plot_GTdict[idx] = plot_text
        # plot_interval_dict[l] = idx
        # Need to handle these two... plot_GTdict is used to get the ground truth plot for observation_dict
        prefact_list, postfact_list, staticfact_list = self.preFact, self.postFact, self.staticFact
        for fact_dict in prefact_list:
            fact, l, r = fact_dict["fact"], fact_dict["interval"][0], fact_dict["interval"][1]
            contradictDetector.interval_insert(fact, l, r, idx, plot_text, isPreFact=True)
        for fact_dict in postfact_list:
            fact, l, r = fact_dict["fact"], fact_dict["interval"][0], fact_dict["interval"][1]
            contradictDetector.interval_insert(fact, l, r, idx, plot_text, isPreFact=False)
        for fact_dict in staticfact_list:
            fact, l, r = fact_dict["fact"], fact_dict["interval_l"][0], fact_dict["interval_l"][1]
            contradictDetector.interval_insert(fact, l, r, idx, plot_text, isPreFact=True)
            l, r = fact_dict["interval_r"][0], fact_dict["interval_r"][1]
            contradictDetector.interval_insert(fact, l, r, idx, plot_text, isPreFact=False)

    def retrieve_bySimilarity(self, num_candidate = 3):
        # Return: list of (local_plot_idx, local_plot_text, similarity)
        # code from dataset_anno/injection_dataset.py
        curr_text, curr_idx = self.outlineItem.main_plot, self.outlineItem.idx
        if self.use_fullPlot:
            curr_text = self.outlineItem.full_plot()
        _, outlineItem_list = self.root_outline.get_partial_outline(curr_idx)
        print(outlineItem_list)
        print(curr_text, curr_idx)
        embedding_current = get_embedding_contriever(curr_text)
        plot_list = []
        from tqdm import trange
        for i in trange(len(outlineItem_list)):
            local_plot_text = outlineItem_list[i].main_plot
            if self.use_fullPlot:
                local_plot_text = outlineItem_list[i].full_plot()
            local_plot_idx = outlineItem_list[i].idx
            embedding = get_embedding_contriever(local_plot_text)
            similarity = similarity_from_embedding(embedding_current, embedding)
            plot_list.append((local_plot_idx, local_plot_text, similarity))
        # Step 3: inject the relevant plots into the current plot
        plot_list = sorted(plot_list, key=lambda x: x[2], reverse=True)
        plot_list = plot_list[:num_candidate]
        # rerank by idx
        plot_list = sorted(plot_list, key=lambda x: x[0])
        self.retrieveCache_similarity = plot_list
        print("")
        print("Plot List:")
        print(plot_list)
        print("")
        return plot_list

    def rewrite_byPlotInject(self):
        if self.rewriteCache_similarity != None:
            return self.rewriteCache_similarity
        if self.retrieveCache_similarity == None:
            plot_list = self.retrieve_bySimilarity()
        else:
            plot_list = self.retrieveCache_similarity
        plot_idx, plot_text = self.outlineItem.idx, self.outlineItem.main_plot
        if self.use_fullPlot:
            plot_text = self.outlineItem.full_plot()
        boundary_event = self.outlineItem.has_boundry_event
        if boundary_event:
            meta_outline = """Main plot: [TODO]
Characters: [TODO]
Begin event: [TODO]
End event: [TODO]"""
        else:
            meta_outline = """Main plot: [TODO]
Characters: [TODO]"""
        prompt = f"""Below is a Current Plot Point. Please rewrite it to make it more consistent with the given Existing Plot Points, taking into account that the outline is structured as a tree. In this tree-like structure, individual points such as 1.1, 1.2, and 1.3 are child nodes of plot point 1. Retain as much of the original content as possible, and maintain clarity and coherence.

Plot {self.outlineItem}
Existing Plot Points:
"""
        for i in range(len(plot_list)):
            prompt += f"Plot Point {plot_list[i][0]}: {plot_list[i][1]}\n"
        prompt += f"""
Rewrited Current Plot Point {plot_idx}: 
{meta_outline}"""
        # print(prompt)
        # Step 4: generate the rewritten plot
        answer = self.model_rewrite([prompt])[0]
        print("##############################################################################################################")
        print(prompt)
        print("*"*100)
        print(answer)
        print("##############################################################################################################")
        # Never change the original plot automatically
        self.rewriteCache_similarity = answer
        # Then there will be a new outlineItem
        return answer

    def retrieve_byContradict(self, num_candidate = 3):
        # Return: (local_plot_idx, local_plot_text, local_fact, isPrefact, interval, NLI_score, curr_fact)
        # The code is from expand_outline_injection@injection/outline_generation_BFS.py
        # If the return length is 0, then it means there is no contradict
        # If the candidate is larger than 3, then return them by randomly / rank them by NLI score?
        if self.observation_dict == None:
            raise Exception("Please run fact_check first!")
        observation_dict = self.observation_dict # Then we can get all result from this dict
        # {"idx", "plot", "error",
        #  "facts":[
        #  {"error fact",
        #  "error fact interval",
        #  "exist fact":[{"fact", "fact interval", "plot", "plot id", isPrefact, nli_score}]
        #  ]}]
        #  }
        print("Now is retrieve by fact contradict!!!!!!!!!")
        print(observation_dict)
        plot_text, plot_idx = self.outlineItem.main_plot, self.outlineItem.idx
        if self.use_fullPlot:
            plot_text = self.outlineItem.full_plot()
        errorFact_list = []
        for fact in observation_dict['facts']:
            for exist_fact in fact['exist fact']:
                local_plot_idx, local_plot_text = exist_fact['plot id'], exist_fact['plot']
                local_fact, isPrefact = exist_fact['fact'], local_plot_idx < plot_idx
                interval = fact['error fact interval']
                NLI_score = exist_fact['nli_score']
                curr_fact = fact['error fact']
                errorFact_list.append((local_plot_idx, local_plot_text, local_fact, isPrefact, interval, NLI_score, curr_fact))
        # rerank by NLI score
        errorFact_list = sorted(errorFact_list, key=lambda x: x[5], reverse=True)
        errorFact_list = errorFact_list[:num_candidate]
        # rerank by idx
        errorFact_list = sorted(errorFact_list, key=lambda x: x[0])
        self.retrieveCache_contradict = errorFact_list
        print("")
        print("Error Fact List:")
        print(errorFact_list)
        print("")
        return errorFact_list

    def rewrite_byFactInject(self):
        if self.rewriteCache_contradict != None:
            return self.rewriteCache_contradict
        # The code is at injection/plot_injection.py
        errorFact_list = self.retrieve_byContradict() if self.retrieveCache_contradict == None else self.retrieveCache_contradict

        boundary_event = self.outlineItem.has_boundry_event
        if boundary_event:
            meta_outline = """Main plot: [TODO]
Characters: [TODO]
Begin event: [TODO]
End event: [TODO]"""
        else:
            meta_outline = """Main plot: [TODO]
Characters: [TODO]"""

        # local_plot_idx, local_plot_text, local_fact, isPrefact, interval, NLI_score, curr_fact
        prompt = f"""Below is a Plot Point which contradicts one or more Existing Facts. Please rewrite the Plot Point to align with all Existing Facts, while keeping as much of the original information as possible and maintaining a clear and concise description. 

Plot {self.outlineItem}
Existing Facts:"""
        length = len(errorFact_list)
        # TODO, to be fixed by current retrieve result
        # Maybe also use a stub outline to test that
        for idx in range(length):
            local_plot_idx, local_plot_text, local_fact, isPrefact, interval, NLI_score, curr_fact = errorFact_list[idx]
            fact = local_fact
            if isPrefact:  # Which is a contradicted prefact
                fact = "Before the Plot Point, " + fact
            else:
                fact = "After the Plot Point, " + fact
            prompt += f"\n{fact}"
        prompt += "\n\nRewritten Plot Point:\n" + meta_outline + "\n"
        answer = self.model_rewrite([prompt])[0]
        self.rewriteCache_contradict = answer
        print("#"*100)
        print("Contradict Based Rewrite:")
        print(prompt)
        print("*"*100)
        print(answer)
        print("#"*100)
        return answer

    def rewrite_byPlotFactInject(self):
        if self.rewriteCache_mix != None:
            return self.rewriteCache_mix
        plot_list = self.retrieve_bySimilarity() if self.retrieveCache_similarity == None else self.retrieveCache_similarity
        errorFact_list = self.retrieve_byContradict() if self.retrieveCache_contradict == None else self.retrieveCache_contradict
        plot_text, plot_idx = self.outlineItem.main_plot, self.outlineItem.idx
        # Todo: combine the two rewrite method
        # Other benchmark, it is easier to extend it after refactor the code
        boundary_event = self.outlineItem.has_boundry_event
        if boundary_event:
            meta_outline = """Main plot: [TODO]
Characters: [TODO]
Begin event: [TODO]
End event: [TODO]"""
        else:
            meta_outline = """Main plot: [TODO]
Characters: [TODO]"""
        prompt = f"""Below is a Current Plot Point. Please rewrite it to make it more consistent with the given Existing Plot Points and Existing Facts, taking into account that the outline is structured as a tree. In this tree-like structure, individual points such as 1.1, 1.2, and 1.3 are child nodes of plot point 1. Retain as much of the original content as possible, and maintain clarity and coherence.

Plot {self.outlineItem}
Existing Plot Points:
"""
        for i in range(len(plot_list)):
            prompt += f"Plot Point {plot_list[i][0]}: {plot_list[i][1]}\n"
        prompt += "Existing Facts:\n"
        length = len(errorFact_list)

        for idx in range(length):
            local_plot_idx, local_plot_text, local_fact, isPrefact, interval, NLI_score, curr_fact = errorFact_list[idx]
            fact = local_fact
            if isPrefact:  # Which is a contradicted prefact
                fact = "Before the Plot Point, " + fact
            else:
                fact = "After the Plot Point, " + fact
            prompt += f"\n{fact}"

        prompt += f"\n\nRewrited Current Plot Point {plot_idx}:\n{meta_outline}\n"
        # print(prompt)
        # Step 4: generate the rewritten plot
        answer = self.model_rewrite([prompt])[0]
        print("#####################################################################################################")
        print(prompt)
        print("*" * 100)
        print(answer)
        print("#####################################################################################################")
        # Never change the original plot automatically
        self.rewriteCache_mix = answer
        return answer



    def rewrite_bySomethingElse(self):
        raise Exception("Not implemented yet!")
        # Other benchmark, it is easier to extend it after refactor the code

    def fact_inject(self, method = "fact", keep_both = False):
        # method: "fact", "plot"
        # Return: a outlineItem that is rewrited,
        #   if there is no contradiction, return False
        #   if the contradiction can't be solved, return True
        # In this function we may need generate a new OutlineItemStateChecker
        flag = True
        counter = 0
        curr_stateChecker = self
        while flag:
            # Step 1: generate the rewritten plot
            if keep_both:
                curr_stateChecker.retrieve_bySimilarity()
                curr_stateChecker.retrieve_byContradict()
            if method == "fact":
                answer = curr_stateChecker.rewrite_byFactInject()
            elif method == "plot":
                answer = curr_stateChecker.rewrite_byPlotInject()
            elif method == "plot_fact":
                answer = curr_stateChecker.rewrite_byPlotFactInject()
            else:
                answer = curr_stateChecker.rewrite_bySomethingElse()
            print(answer)
            answer = f"{self.outlineItem.idx}\n{answer}"
            # raise Exception("Please check the correctness of the answer!")
            # # To be correct: we rewrite the full outlineItem, instead of rewrite the main_plot
            # # Copy the old outlineItem and just modify the new one?
            # import copy
            # new_outlineItem = copy.deepcopy(self.outlineItem)
            # new_outlineItem.main_plot = answer
            new_outlineItem = OutlineItem(answer)
            print("&"*100)
            print("New Outline Item:")
            print(new_outlineItem)
            print("&"*100)
            # Step 2: make a new state checker
            new_stateChecker = OutlineItemStateChecker(new_outlineItem, self.l, self.r,
                                                       self.contradictDetector, self.root_outline,
                                                       self.model_rewrite, self.use_fullPlot)

            # then if there is a contradict, back to step 1, otherwise update the status
            new_stateChecker.fact_decompose()
            if new_stateChecker.fact_check():
                counter += 1
                if counter > 3:
                    # rewrite at most 3 times
                    print("!"*100)
                    print("Can't solve the contradiction!")
                    print("!"*100)
                    return new_stateChecker
                curr_stateChecker = new_stateChecker
                continue
            return new_stateChecker