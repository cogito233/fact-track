import pandas as pd
import os
BASE_PATH = os.environ["BASE_PATH"]

import sys
sys.path.append(f"{BASE_PATH}/fact-track/fact-track/utils")
from gpt_api import load_model2classification, openai_UnitContradictCheck
from huggingface_api import get_embedding_contriever, similarity_from_embedding, huggingface_UnitContradictScore

def huggingface_UnitContradictCheck(fact1, fact2, nli_score = None):
    return nli_score > 0.2359 # 3% point

def huggingface_UnitBlockCheck(fact1, fact2, nli_score = None):
    # return nli_score > 0.5 # 2% point
    return nli_score > 0.8 # ?% point

# This contradict Detector is used to BFS because it can contain the global status instead only the single time stamp
class ContradictDetector_StatusPersistence(object):
    """
    Different Operations and their Constant Cost: (From low to High), we can ensure that the low constant cost operation can have a higher complexity.
    0. Sort the factList: O(nlogn)
    1. Iterate each fact in the factList: O(n)
    2. Check the similarity between two facts: O(n) # But break when it is invalid interval
    3. Check the contradiction by NLI model between two facts: O(n) or O(logn)
        # In contradict check, is O(n) because of reranking, in block check, is O(logn) because of binary lifting
    4. Check the contradiction by GPT query between two facts: O(1) or O(logn)
        # O(max_contradict_query) in contradict check, O(binary_lifting_times * max_contradict_query) in block check
    """
    def __init__(self, model_name_decompose = "gpt-3.5-turbo", model_name_contradict = "gpt-4",
                 log_file_name = "contradict_log.log", contradict_file_name = "contradict_list",
                 similarity_threshold = 0.5, nli_threshold = 0.2359, same_threshold = 0.95):
        # OpenAI API
        if model_name_decompose == "llama2":
            from llama_api import load_model2generation
            self.model_decompose = load_model2generation()
        elif model_name_decompose == "llama2-7B-chat":
            from llama_api_vLLM import load_deterministic_llama2
            self.model_decompose = load_deterministic_llama2("7B-chat")
        elif model_name_decompose == "llama2-13B-chat":
            from llama_api_vLLM import load_deterministic_llama2
            self.model_decompose = load_deterministic_llama2("13B-chat")
        else:
            self.model_decompose = load_model2classification(model_name_decompose)
        if model_name_contradict != "huggingface":
            self.model_contradict = load_model2classification(model_name_contradict)
        else:
            self.model_contradict = load_model2classification("gpt-3.5-turbo")
        self.model_name_contradict = model_name_contradict

        # Inner Storage
        self.prefactList = [] # list of (fact, embedding, l, r, sourcePlot_idx), sorted by l
        self.postfactList = [] # list of (fact, embedding, l, r, sourcePlot_idx), sorted by r

        # Threshold
        self.similarity_threshold = similarity_threshold
        self.nli_threshold = nli_threshold
        self.local_nli_threshold = nli_threshold
        self.local_nli_counter = 0
        self.max_local_nli_counter = 1
        self.max_resident_candidate = 3
        self.same_threshold = same_threshold # That means two fact told the same thing.
        # self.tradition_NLI_threshold = tradition_NLI_threshold
        self.max_contradict_query = 5
        self.epsilon = 1e-6

        # Output File Name and Content
        self.log_file_name = log_file_name
        self.factContradict_file_name = f"{contradict_file_name}_factContradict.csv"
        self.contradict_list = {
            "fact1": [],
            "fact2": [],
            "plot1_idx": [],
            "plot2_idx": [],
            "similarity": [],
            "nli_score": [],
            "label": [],
            "where": [],
            "is_query": [],
        }
        self.fact_file_name = f"{contradict_file_name}_factplot.csv"
        self.fact_list = {
            "plot": [],
            "fact": [],
            "type": [],
        }
        # list of {fact1, fact2, plot1_idx, plot2_idx, similarity, contradict_score, label}
        # Plot Pair Contradict is not needed because we can get it from factContradict
        # self.plotContradict_file_name = f"{contradict_file_name}_plotContradict.csv"
        # self.plotContradict_list = []
        # list of {plot1, plot2, plot1_idx, plot2_idx}

        with open(self.log_file_name, "w") as f:
            # Put the meta information of the model here.
            # time stamp
            f.write("Current Time Stamp: \n")
            import datetime
            f.write(str(datetime.datetime.now()) + "\n")
            # model name
            f.write(f"Model Name Decompose: {model_name_decompose}\n")
            # model name
            f.write(f"Model Name Contradict: {model_name_contradict}\n")
            # contradict method name
            f.write(f"Contradict Detector: {self.model_name_contradict}\n")
            # similarity threshold and nli threshold
            f.write(f"Similarity Threshold: {self.similarity_threshold}\n")
            f.write(f"NLI Threshold: {self.nli_threshold}\n")

    def end_log(self):
        # log the usage of the model.
        prefactList = sorted(self.prefactList, key = lambda x: x[3]) # sort by r
        postfactList = sorted(self.postfactList, key = lambda x: x[2]) # sort by l
        # print("Final Pre-fact List: ")
        # for fact in prefactList:
        #     print(f"Fact: {fact[0]}, Interval:({fact[2]}, {fact[3]}], Source Plot Idx: {fact[4]}")
        # print("Final Post-fact List: ")
        # for fact in postfactList:
        #     print(f"Fact: {fact[0]}, Interval:[{fact[2]}, {fact[3]}), Source Plot Idx: {fact[4]}")
        # print("Model decompose usage: " + str(self.model_decompose.summarize))
        # print("Model contradict usage: " + str(self.model_contradict.summarize))
        with open(self.log_file_name, "a") as f:
            # f.write("Model decompose usage: " + str(self.model_decompose.summarize) + "\n")
            # f.write("Model contradict usage: " + str(self.model_contradict.summarize) + "\n")
            f.write("Final Pre-fact List: \n")
            for fact in prefactList:
                f.write(f"Fact: {fact[0]}, Interval:({fact[2]}, {fact[3]}], Source Plot Idx: {fact[4]}\n")
            f.write("Final Post-fact List: \n")
            for fact in postfactList:
                f.write(f"Fact: {fact[0]}, Interval:[{fact[2]}, {fact[3]}), Source Plot Idx: {fact[4]}\n")
        # save the contradict fact list to the file.
        df = pd.DataFrame(self.contradict_list)
        # print(df.head())
        df.to_csv(self.factContradict_file_name)
        df_fact = pd.DataFrame(self.fact_list)
        df_fact.to_csv(self.fact_file_name)

    def clean_all_fact(self):
        self.prefactList = []
        self.postfactList = []

    def fact_decompose(self, plot_point_text):
            """
            Decompose a plot point into a list of facts.
            pre-facts: the facts that are valid before the plot event.
            post-facts: the facts that are valid after the plot event.
            static facts: the facts that are valid before and after the plot event.
            """

            def get_factList(str):
                factList = str.split("\n")
                factList = [fact[2:] for fact in factList if len(fact) > 3 and "Let me know if you" not in fact]
                factList = [fact for fact in factList if "event):" not in fact]
                return factList

            prompt = f"""Deconstruct the given plot point into atomic facts, considering facts valid until before the plot event (pre-facts), facts valid starting after the plot event (post-facts), and facts that remain valid throughout the event (static facts). For pre-facts, identify the conditions that are present before the event, but change as a result of it. For post-facts, identify the conditions that are valid after the event, which are essentially the transformed versions of the corresponding pre-facts. Static facts are the conditions that remain true throughout the event. Please be sure to present facts as assertive statements, rather than speculative or suggestive ones. 

Plot Point: {plot_point_text} 

Pre-Facts: 
[pre-facts]

Post-Facts: 
[post-facts]

Static Facts:
[static facts] 
"""
            # PreFacts, PostFacts, StaticFacts = [], [], []
            answer = self.model_decompose([prompt])[0]
            print(answer)
            try:
                preStr = answer.split("Pre-Facts")[1].split("Post-Facts")[0]
                postStr = answer.split("Post-Facts")[1].split("Static Facts")[0]
                staticStr = answer.split("Static Facts")[1]
                PreFacts = get_factList(preStr)
                PostFacts = get_factList(postStr)
                StaticFacts = get_factList(staticStr)
            except:
                print("Decompose Error!")
                print(answer)
                PreFacts, PostFacts, StaticFacts = [], [], []
                # Also put into the log file
                with open(self.log_file_name, "a") as f:
                    f.write("Decompose Error!\n")
                    f.write("Plot Point: " + plot_point_text + "\n")
                    f.write("Answer: " + answer + "\n")

            for fact in PreFacts:
                # Move the "1." stuff.
                self.fact_list['fact'].append(fact)
                self.fact_list['type'].append("pre")
                self.fact_list['plot'].append(plot_point_text)
            for fact in PostFacts:
                self.fact_list['fact'].append(fact)
                self.fact_list['type'].append("post")
                self.fact_list['plot'].append(plot_point_text)
            for fact in StaticFacts:
                self.fact_list['fact'].append(fact)
                self.fact_list['type'].append("static")
                self.fact_list['plot'].append(plot_point_text)

            # log the print to the log file.
            with open(self.log_file_name, "a") as f:
                f.write("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n")
                f.write("Fact decompose done!\n")
                f.write("Plot Point: " + plot_point_text + "\n")
                f.write("Pre-facts:" + str(PreFacts) + "\n")
                f.write("Post-facts:" + str(PostFacts) + "\n")
                f.write("Static facts:" + str(StaticFacts) + "\n")

            return PreFacts, PostFacts, StaticFacts

    def contradict_check(self, fact1, fact2, nli_score = None):
        if nli_score is None:
            nli_score = huggingface_UnitContradictScore(fact1, fact2)
        if nli_score > self.nli_threshold:
            # return True
            if self.model_name_contradict == 'gpt-4':
                return openai_UnitContradictCheck(fact1, fact2, model = self.model_contradict), nli_score
            elif self.model_name_contradict == 'gpt-3.5-turbo':
                return nli_score > 0.9 or openai_UnitContradictCheck(fact1, fact2, model = self.model_contradict), nli_score
                # Esemble the two model
            elif self.model_name_contradict == 'huggingface':
                return huggingface_UnitContradictCheck(fact1, fact2, nli_score = nli_score), nli_score
                # return nli_score > 0.1, nli_score # TODO, change the threshold
            else:
                raise Exception("Invalid model name!")
        else:
            return False, nli_score

    def block_check_reset(self):
        self.local_nli_threshold = self.nli_threshold
        self.local_nli_counter = 0

    def block_check(self, fact1, fact2, fact1_embedding, fact2_embedding):
        # used for check whether a fact block another fact
        # Return is (flag, similarity, nli_score, is_query)
        similarity = similarity_from_embedding(fact1_embedding, fact2_embedding)
        if similarity > self.same_threshold:
            nli_score = huggingface_UnitContradictScore(fact1, fact2)
            return True, similarity, nli_score, False
        elif similarity < self.similarity_threshold:
            return False, similarity, -1, False
        else:
            nli_score = huggingface_UnitContradictScore(fact1, fact2)
            if nli_score > self.local_nli_threshold:
                if self.model_name_contradict == 'gpt-4':
                    self.local_nli_counter += 1
                    if self.local_nli_counter >= self.max_local_nli_counter:
                        self.local_nli_threshold = self.local_nli_threshold * 2
                        self.local_nli_counter = 0
                    return openai_UnitContradictCheck(fact1, fact2, model=self.model_contradict), similarity, nli_score, True
                elif self.model_name_contradict == 'gpt-3.5-turbo':
                    if nli_score < 0.9:
                        self.local_nli_counter += 1
                        if self.local_nli_counter >= self.max_local_nli_counter:
                            self.local_nli_threshold = self.local_nli_threshold * 2
                            self.local_nli_counter = 0
                    # Esemble the two model
                    return nli_score > 0.9 or openai_UnitContradictCheck(fact1, fact2, model=self.model_contradict), similarity, nli_score, nli_score < 0.9
                elif self.model_name_contradict == 'huggingface':
                    return huggingface_UnitBlockCheck(fact1, fact2, nli_score = nli_score), similarity, nli_score, False
                else:
                    raise Exception("Invalid model name!")
            else:
                return False, similarity, nli_score, False

    def standardize_log(self, fact1, fact2, plot1_idx, plot2_idx, similarity, nli_score, location = "checkBlock", label = None,
                        source_interval = None, target_interval = None, is_query = False):
        # TODO update the all execute of contradict_check and block_check
        with open(self.log_file_name, "a") as f:
            f.write("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n")
            if location == "checkBlock":
                f.write(f"Now is Check Block, the current fact is from {plot1_idx} and the fact in world model is from {plot2_idx}, the label is {label}\n")
                if label == 1:
                    f.write("The current fact is blocked by the fact in world model!\n")
                self.contradict_list["where"].append("checkBlock")
            elif location == "checkContradict":
                if label == None:
                    raise Exception("Invalid label!")
                f.write(f"Now is Check Contradict, the candidate pair is from {plot1_idx} and {plot2_idx}, and the label is {label}\n")
                if label == 1:
                    f.write("The two fact is contradict!\n")
                self.contradict_list["where"].append("checkContradict")
            elif location == "updateBlock":
                f.write(f"Now is update the world model, the current fact is from {plot1_idx} and the fact in world model is from {plot2_idx}, the label is {label}\n")
                if label == 1:
                    f.write("The current fact is block the fact in world model!\n")
                self.contradict_list["where"].append("updateBlock")
            else:
                raise Exception("Invalid location name!")
            self.contradict_list["fact1"].append(fact1)
            self.contradict_list["fact2"].append(fact2)
            self.contradict_list["plot1_idx"].append(plot1_idx)
            self.contradict_list["plot2_idx"].append(plot2_idx)
            self.contradict_list["similarity"].append(similarity)
            self.contradict_list["nli_score"].append(nli_score)
            self.contradict_list["label"].append(label)
            self.contradict_list["is_query"].append(is_query)
            f.write("Source Fact: " + fact1 + "\n")
            f.write("Target Fact: " + fact2 + "\n")
            if source_interval[0] < source_interval[1]:
                f.write(f"Source Interval: [{source_interval[0]}, {source_interval[1]})")
            else:
                f.write(f"Source Interval: ({source_interval[1]}, {source_interval[0]}]")
            f.write("\n")
            if target_interval[0] < target_interval[1]:
                f.write(f"Target Interval: [{target_interval[0]}, {target_interval[1]})")
            else:
                f.write(f"Target Interval: ({target_interval[1]}, {target_interval[0]}]")
            f.write("\n")
            f.write("Similarity: " + str(similarity) + " NLI_score:" + str(nli_score) + "\n")

    # TODO: add plot_idx and log
    def interval_blockCheck(self, fact, base, plot_idx, isPreFact = True):
        # Find the valid interval of the fact
        # If the fact is a pre-fact, then the initiate interval is (-inf, base]
        # If the fact is a post-fact, then the initiate interval is [base, +inf)
        # Check whether the interval is blocked by the existing facts with the same type
        # Blocked by the most nearest fact that is contradict or have the same meaning
        self.block_check_reset()
        candidateFactList = []
        fact_embedding = get_embedding_contriever(fact)
        if isPreFact:
            l, r = -1e9, base
            # Sort the pre-fact from larger to smaller by the base
            prefactList = sorted(self.prefactList, key=lambda x: x[3], reverse=True)
            for fact_inWorld in prefactList: # From larger to smaller
                #if l < fact_inWorld[3]: # l ls smaller than the base of the fact_inWorld
                if base > fact_inWorld[3]: # current base is larger than the base of the fact_inWorld
                    block_flag, similarity, nli_score, is_query = self.block_check(fact, fact_inWorld[0], fact_embedding, fact_inWorld[1])
                    if similarity != None and similarity > self.similarity_threshold and nli_score > self.nli_threshold:
                        self.standardize_log(fact, fact_inWorld[0], plot_idx, fact_inWorld[-2], similarity, nli_score, label=block_flag,
                                             location="checkBlock", source_interval=[r, l], target_interval=[fact_inWorld[3], fact_inWorld[2]], is_query=is_query)
                    if block_flag:
                        # Return on the first fact that block the interval
                        return fact_inWorld[3] + self.epsilon, r
                    if not is_query: # That means the fact is not checked by GPT4 Query
                        candidateFactList.append([fact_inWorld, similarity, nli_score])
        else:
            l, r = base, 1e9
            # Sort the post-fact from smaller to larger by the base
            postfactList = sorted(self.postfactList, key=lambda x: x[2])
            for fact_inWorld in postfactList: # From smaller to larger
                if base < fact_inWorld[2]: # current base is smaller than the base of the fact_inWorld
                    block_flag, similarity, nli_score, is_query = self.block_check(fact, fact_inWorld[0], fact_embedding, fact_inWorld[1])
                    if similarity != None and similarity > self.similarity_threshold and nli_score > self.nli_threshold:
                        self.standardize_log(fact, fact_inWorld[0], plot_idx, fact_inWorld[-2], similarity, nli_score, label = block_flag,
                                             location="checkBlock", source_interval=[l, r], target_interval=[fact_inWorld[2], fact_inWorld[3]], is_query=is_query)
                    if block_flag:
                        return l, fact_inWorld[2] - self.epsilon
                    if not is_query: # That means the fact is not checked by GPT4 Query
                        # That means the fact is not checked by GPT4 Query
                        candidateFactList.append([fact_inWorld, similarity, nli_score])
        # Check the candidate fact list
        candidateFactList = sorted(candidateFactList, key=lambda x: x[1], reverse=True)
        count = 0
        for candidateFact in candidateFactList:
            count += 1
            self.block_check_reset()
            block_flag, similarity, nli_score, is_query = self.block_check(fact, candidateFact[0][0], fact_embedding, candidateFact[0][1])
            if similarity != None and similarity > self.similarity_threshold and nli_score > self.nli_threshold:
                self.standardize_log(fact, candidateFact[0][0], plot_idx, candidateFact[0][-1], similarity, nli_score,
                                     label=block_flag,
                                     location="checkBlock", source_interval=[l, r] if isPreFact else [r, l],
                                     target_interval=[candidateFact[0][2], candidateFact[0][3]] if isPreFact else [candidateFact[0][3], candidateFact[0][2]], is_query=is_query)
            if block_flag:
                if isPreFact:
                    return candidateFact[0][3] + self.epsilon, r
                else:
                    return l, candidateFact[0][2] - self.epsilon
            if count >= self.max_resident_candidate:
                break
        return l, r

    # TODO: add plot_idx and log
    def interval_contradictCheck(self, fact, l, r, plot_idx, isPreFact = True): # PreFact means begin point is r and end point is l
        # Check if the current fact is contradict with any existing fact with the different type
        # for PreFact, the interval is (l, r], for PostFact, the interval is [l, r)
        # (l, r] and [l1, r1) the overlap condition is l<l1<r<r1
        # check the contradict only if the interval is overlapped
        # TODO: What if only one fact overlap with the interval? Does it can happen?
        def is_overlap(l, r, l1, r1, isPreFact):
            #print("is_overlap")
            #print(l,r, l1,r1, isPreFact)
            if isPreFact:
                return l < l1 and l1 < r and r < r1
            else:
                return l1 < l and l < r1 and r1 < r
        fact_embedding = get_embedding_contriever(fact)
        candidateFactList = []
        #print(len(self.prefactList), len(self.postfactList))
        #print("#####################################")
        for fact_inWorld in self.postfactList if isPreFact else self.prefactList:
            if is_overlap(l, r, fact_inWorld[2], fact_inWorld[3], isPreFact):
                similarity = similarity_from_embedding(fact_embedding, fact_inWorld[1])
                if similarity < self.similarity_threshold:
                    # If the two facts are not similar, then they are not contradict
                    continue
                nli_score = huggingface_UnitContradictScore(fact, fact_inWorld[0])
                # plot_idx, plot_text = fact_inWorld[-2], fact_inWorld[-1]
                if nli_score > self.nli_threshold:
                    candidateFactList.append([fact_inWorld, nli_score, similarity])
        # Sort the candidateFactList by the nli_score from larger to smaller
        candidateFactList = sorted(candidateFactList, key=lambda x: x[1], reverse=True)
        flag = False
        meta_info = [] # [{"fact", "fact interval", "plot", "plot id"}]
        for i in range(len(candidateFactList)):
            if i > self.max_contradict_query: # TODO: maybe the block check and contradict check can use different max_contradict_query
                break
            fact_inWorld = candidateFactList[i][0]
            isContradict, nli_score = self.contradict_check(fact, fact_inWorld[0], nli_score=candidateFactList[i][1])
            flag = flag or isContradict
            # Add the information about whether the fact is contradict with the current fact
            self.standardize_log(fact, fact_inWorld[0], plot_idx, fact_inWorld[-2], candidateFactList[i][2], candidateFactList[i][1],
                                 location="checkContradict", label=isContradict,
                                 source_interval=[l, r] if isPreFact else [r, l],
                                 target_interval=[fact_inWorld[2], fact_inWorld[3]] if isPreFact else [fact_inWorld[3], fact_inWorld[2]], is_query=True)
            if isContradict:
                meta_info.append({"fact": fact_inWorld[0], "fact interval": [fact_inWorld[2], fact_inWorld[3]],
                                  "plot": fact_inWorld[-1], "plot id": fact_inWorld[-2], "isPreFact": isPreFact,
                                  "nli_score": candidateFactList[i][1]})
                # print(meta_info)
                # exit(0)
                # TODO: need to add NLI score?
        return flag, meta_info

    # TODO: add plot_idx and log
    def interval_insert(self, fact, l, r, plot_idx, plot_text, isPreFact = True):
        # Check all decendent facts that are blocked by the current fact, update their interval
        self.block_check_reset()
        candidateFactList = []
        fact_embedding = get_embedding_contriever(fact)
        if isPreFact:
            # Update all fact status
            # Sort prefactList by the l value from smaller to larger
            prefactList = sorted(self.prefactList, key=lambda x: x[2])
            for fact_inWorld in prefactList:
                if fact_inWorld[3] <= r: # Now base is l
                    continue
                if fact_inWorld[2] < r: # Now base is r
                    block_flag, similarity, nli_score, is_query = self.block_check(fact, fact_inWorld[0], fact_embedding, fact_inWorld[1])
                    if similarity != None and similarity > self.similarity_threshold and nli_score > self.nli_threshold:
                        self.standardize_log(fact, fact_inWorld[0], plot_idx, fact_inWorld[-2], similarity, nli_score,
                                             label=block_flag,
                                             location="updateBlock", source_interval=[r, l],
                                             target_interval=[fact_inWorld[3], fact_inWorld[2]], is_query=is_query)
                    if block_flag:
                        fact_inWorld[2] = r + self.epsilon # update the end point when blocked
                    if not is_query: # That means the fact is not checked by GPT4 Query
                        candidateFactList.append([fact_inWorld, nli_score, similarity])
                else:
                    # If the fact_inWorld is not overlapped with the current fact, then the following facts are not overlapped because of the sort.
                    break
            fact_insert = [fact, fact_embedding, l, r, plot_idx, plot_text]
            self.prefactList.append(fact_insert)
        else:
            # Sort postfactList by the r value from larger to smaller
            postfactList = sorted(self.postfactList, key=lambda x: x[3], reverse=True)
            for fact_inWorld in postfactList:
                if fact_inWorld[2] >= l: # Now base is r
                    continue
                if fact_inWorld[3] > l: # Now base is l
                    block_flag, similarity, nli_score, is_query = self.block_check(fact, fact_inWorld[0], fact_embedding, fact_inWorld[1])
                    if similarity != None and similarity > self.similarity_threshold and nli_score > self.nli_threshold:
                        self.standardize_log(fact, fact_inWorld[0], plot_idx, fact_inWorld[-2], similarity, nli_score,
                                             label=block_flag,
                                             location="updateBlock", source_interval=[l, r],
                                             target_interval=[fact_inWorld[2], fact_inWorld[3]], is_query=is_query)
                    if block_flag:
                        fact_inWorld[3] = l - self.epsilon # update the begin point when blocked
                    if not is_query: # That means the fact is not checked by GPT4 Query
                        candidateFactList.append([fact_inWorld, nli_score, similarity])
                else:
                    break
            fact_insert = [fact, fact_embedding, l, r, plot_idx, plot_text]
            self.postfactList.append(fact_insert)
        candidateFactList = sorted(candidateFactList, key=lambda x: x[1], reverse=True)
        count = 0
        for candidateFact in candidateFactList:
            count += 1
            self.block_check_reset()
            block_flag, similarity, nli_score, is_query = self.block_check(fact, candidateFact[0][0], fact_embedding,
                                                                 candidateFact[0][1])
            if similarity != None and similarity > self.similarity_threshold and nli_score > self.nli_threshold:
                self.standardize_log(fact, candidateFact[0][0], plot_idx, candidateFact[0][-1], similarity, nli_score,
                                        label=block_flag, location="updateBlock", source_interval=[l, r] if isPreFact else [r, l],
                                        target_interval=[candidateFact[0][2], candidateFact[0][3]] if isPreFact else [candidateFact[0][3], candidateFact[0][2]], is_query=is_query)
            if block_flag:
                if isPreFact:
                    candidateFact[0][2] = r + self.epsilon
                else:
                    candidateFact[0][3] = l - self.epsilon
            if count >= self.max_resident_candidate:
                break

    def add_fact(self, fact, base, plot_idx, isPreFact = True):
        # return is whether generate a contradict
        l, r = self.interval_blockCheck(fact, base, plot_idx, isPreFact)
        isContradict, meta_info = self.interval_contradictCheck(fact, l, r, plot_idx, isPreFact)
        # print("Is Contradict?", isContradict)
        # print("l, r", l, r)
        # print("fact: ", fact)
        self.interval_insert(fact, l, r, plot_idx, isPreFact = isPreFact)
        return isContradict

    def add_facts(self, facts, base, plot_idx, isPreFact = True):
        flag = False
        for fact in facts:
            flag_sub = self.add_fact(fact, base, plot_idx, isPreFact = isPreFact)
            flag = flag or flag_sub
        return flag

    def add_plotDecomposition(self, prefact, postfact, staticfact, l, r, plot_idx):
        flag = False
        flag = self.add_facts(prefact, l, plot_idx, isPreFact=True) or flag
        flag = self.add_facts(postfact, r, plot_idx, isPreFact=False) or flag
        # How to handle with the static fact is depends on the static fact type
        # To Be Determined: How to handle with the static fact, currently, we just simply take it as pre-fact plus post-fact
        flag = self.add_facts(staticfact, l, plot_idx, isPreFact=True) or flag
        flag = self.add_facts(staticfact, r, plot_idx, isPreFact=False) or flag
        return flag

def test_pairwise(plot_point_text_1, plot_point_text_2, contradictDetector, pair_idx):
    flag = False

    # check the pre fact and static fact.
    # For plot1, the l and r is 0 and 0.999999
    prefact, postfact, staticfact = contradictDetector.fact_decompose(plot_point_text_1)
    l, r = 0, 0.999999
    plot_idx = f"{pair_idx}.1"
    flag = contradictDetector.add_plotDecomposition(prefact, postfact, staticfact, l, r, plot_idx) or flag

    #return flag
    # For plot2, the l and r is 1 and 1.999999
    prefact, postfact, staticfact = contradictDetector.fact_decompose(plot_point_text_2)
    l, r = 1, 1.999999
    plot_idx = f"{pair_idx}.2"
    flag = contradictDetector.add_plotDecomposition(prefact, postfact, staticfact, l, r, plot_idx) or flag

    # print("Final Fact list: ", contradictDetector.get_all_fact())

    contradictDetector.end_log()
    contradictDetector.clean_all_fact()
    return flag


if __name__ == "__main__":

    contradictDetector = ContradictDetector_StatusPersistence(model_name_decompose = "gpt-4")
    # fact1 = "Max is not in the antique store."
    # fact2 = "Max is in the antique store."
    # contradictDetector.add_fact(fact1, 0.99, "1.1", isPreFact=False)
    # contradictDetector.add_fact(fact2, 1, "1.2", isPreFact=True)
    # exit(0)
    # plot_point_text_1 = "Unaware of their conflict, Samantha is tricked into joining the Dreamweavers to help them against the Nightmares."
    # plot_point_text_2 = "Samantha grows suspicious and finally uncovers the truth about the ongoing conflict between the Dreamweavers and the Nightmares."

    plot_point_text_1 = "Max purchases the book and leaves the antique store."
    plot_point_text_2 = "he Antique Store Owner shows Max a strange book that catches his eye."

    #plot_point_text_1 = "The inventor explains to Ava how the device works and the dangers of traveling through dimensions."
    #plot_point_text_2 = "With her new understanding of the device, Ava cautiously explores its capabilities, fully aware of the dangers associated with dimensional travel."
    flag = test_pairwise(plot_point_text_1, plot_point_text_2, contradictDetector, 1)
    print(flag)
    # contradictDetector.end_log()