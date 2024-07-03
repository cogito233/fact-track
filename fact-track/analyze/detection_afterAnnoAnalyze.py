# 这个程序的目的大概是说, 我们有了一些已经标注出来的pair, 然后想要进行一些analyze
# Code Copy from analyze_code/dataset_anno/detection_dataset_analyze.py

# TODO: rewrite this function to an object
def plot_pair_analyze(outline_idx, mode, plot1_idx, plot2_idx):
    # Return: (1, 2, 3, 4, 5): diffferent status
    # True Contradict, Ignore Contradict, Contain Contradict, Block Contradict, False Contradict
    # Step 1: Load all meta data and prepare the dicts, post-facts and pre-facts
    # Step 1.1: Load meta data
    if plot1_idx > plot2_idx:
        plot1_idx, plot2_idx = plot2_idx, plot1_idx
    # mode = "simple" if outline_idx == "100" else "detail" if outline_idx == "100_detail" else "full"
    outline, _ = load_given_outline(outline_idx, mode = mode)
    outline_dir = f"outline_{outline_idx}"
    if mode != "simple":
        outline_dir += "_" + mode
    dir = f"detection_data/{outline_dir}/"
    df_contradictPair = pd.read_csv(dir + "outline_factContradict.csv")
    df_plot2fact = pd.read_csv(dir + "outline_factplot.csv")
    # print(df_contradictPair.head())
    # print(df_plot2fact.head())
    import pickle
    contradsictDetector_dir = dir + "contradictDetector.pkl"
    with open(contradsictDetector_dir, 'rb') as f:
        contradictDetector = pickle.load(f)
    # Step 1.2 Prepare the dicts and facts
    # plot_idx -> plot_text
    outline_idx2text_dict = outline_idx2text(outline)
    # post fact for plot1, and pre fact for plot2
    plot1_text = outline_idx2text_dict[plot1_idx]
    plot2_text = outline_idx2text_dict[plot2_idx]
    print("plot1_text:", plot1_text)
    print("plot2_text:", plot2_text)
    #print(set(df_plot2fact['type'].tolist()))
    plot1_factDecompose = df_plot2fact[df_plot2fact['plot'] == plot1_text]
    plot2_factDecompose = df_plot2fact[df_plot2fact['plot'] == plot2_text]
    plot1_post_fact = plot1_factDecompose[plot1_factDecompose['type'].isin(['post', 'static'])]['fact'].tolist()
    plot2_pre_fact = plot2_factDecompose[plot2_factDecompose['type'].isin(['pre', 'static'])]['fact'].tolist()
    # (fact, idx) to (l, r) from contradict detector
    prefactDict, postfactDict = {}, {}
    prefactList, postfactList = contradictDetector.prefactList, contradictDetector.postfactList
    # print(prefactList[:3])
    for prefact in prefactList:
        fact_text, fact_embedding, l, r, idx = prefact
        prefactDict[(fact_text, idx)] = (l, r)
        # if idx == plot2_idx:
        #     print((fact_text, idx), (l, r))
    # print(prefactDict[(plot2_pre_fact[0], plot2_idx)])
    # print("="*20)
    for postfact in postfactList:
        fact_text, fact_embedding, l, r, idx = postfact
        postfactDict[(fact_text, idx)] = (l, r)
        # if idx == plot1_idx:
        #     print((fact_text, idx), (l, r))
    # (fact1, fact2, plot1_idx, plot2_idx) to (label, is_query, nli_score)
    fact_pair2label_dict = {}
    for i in range(len(df_contradictPair)):
        item = df_contradictPair.iloc[i]
        fact1, fact2, plot1_idx_local, plot2_idx_local = item['fact1'], item['fact2'], item['plot1_idx'], item['plot2_idx']
        label, is_query, nli_score = item['label'], item['is_query'], item['nli_score']
        fact_pair2label_dict[(fact1, fact2, plot1_idx_local, plot2_idx_local)] = (label, is_query, nli_score)
    # Step 2: Check all the fact pairs
    fact_pairs_list_blocked = [] # (fact1, fact2, status)
    fact_pairs_list_ignore = []
    fact_pairs_list_contain = []
    # print(f"plot1 fact length: {len(plot1_post_fact)}, plot2 fact length: {len(plot2_pre_fact)}")
    for fact1 in plot1_post_fact:
        for fact2 in plot2_pre_fact:
            # print(fact1, fact2)
            l1, r1 = prefactDict[(fact2, plot2_idx)]
            l2, r2 = postfactDict[(fact1, plot1_idx)]
            # print(fact1, fact2, l1, r1, l2, r2)
            if l1 < l2 and l2 < r1 and r1 < r2: # Overlap
                # print("Overlap")
                # check if is in the fact_pair2label_dict, if so, check the label, if not, add into the fact_pairs_list_ignore
                if (fact1, fact2, plot1_idx, plot2_idx) in fact_pair2label_dict:
                    label, is_query, nli_score = fact_pair2label_dict[(fact1, fact2, plot1_idx, plot2_idx)]
                    if label:
                        # return 1 # True Contradict
                        pass
                    if not is_query:
                        fact_pairs_list_ignore.append((fact1, fact2, "ignore"))
                else:
                    fact_pairs_list_ignore.append((fact1, fact2, "ignore"))
            else: # Block
                if r1 < l2:
                    fact_pairs_list_blocked.append((fact1, fact2, "block"))
                else: # Containing relationship, how to do?
                    fact_pairs_list_contain.append((fact1, fact2, "contain"))
    # print(len(fact_pairs_list_blocked), len(fact_pairs_list_ignore), len(fact_pairs_list_contain))
    # Step 3: Classify the status by inference NLI top-5 pairs, # TODO: should we reduce the cost?
    # Step 3.1: Compute all NLI scores
    fact_pairs_list_blocked_nli = []
    fact_pairs_list_ignore_nli = []
    fact_pairs_list_contain_nli = []
    for fact1, fact2, status in fact_pairs_list_blocked:
        nli_score = huggingface_UnitContradictScore(fact1, fact2)
        fact_pairs_list_blocked_nli.append((fact1, fact2, status, nli_score))
    for fact1, fact2, status in fact_pairs_list_ignore:
        nli_score = huggingface_UnitContradictScore(fact1, fact2)
        fact_pairs_list_ignore_nli.append((fact1, fact2, status, nli_score))
    for fact1, fact2, status in fact_pairs_list_contain:
        nli_score = huggingface_UnitContradictScore(fact1, fact2)
        fact_pairs_list_contain_nli.append((fact1, fact2, status, nli_score))
    # Step 3.2: Sort by NLI score
    fact_pairs_list_blocked_nli = sorted(fact_pairs_list_blocked_nli, key=lambda x: x[-1], reverse=True)
    fact_pairs_list_ignore_nli = sorted(fact_pairs_list_ignore_nli, key=lambda x: x[-1], reverse=True)
    fact_pairs_list_contain_nli = sorted(fact_pairs_list_contain_nli, key=lambda x: x[-1], reverse=True)
    # print("="*20+"ignore pairs"+"="*20)
    # for i in range(5):
    #     print(fact_pairs_list_ignore_nli[i])
    # print("="*20+"contain pairs"+"="*20)
    # for i in range(5):
    #     print(fact_pairs_list_contain_nli[i])
    # Step 3.3: Check the top-5 pairs, with the order of ignore, contain, blocked
    model = load_model2classification(model = "gpt-4")
    counter = 0
    for fact1, fact2, status, nli_score in fact_pairs_list_ignore_nli:
        counter += 1
        if counter > 5 or nli_score < 0.02:
            break
        print("Now checking: ", fact1, fact2)
        label = openai_UnitContradictCheck(fact1, fact2, model)
        print("label: ", label)
        if label:
            print("Current Case is Ignore Contradict")
            print("Fact1: ", fact1)
            print("Fact2: ", fact2)
            return 2
    counter = 0
    for fact1, fact2, status, nli_score in fact_pairs_list_contain_nli:
        counter += 1
        if counter > 5 or nli_score < 0.02:
            break
        label = openai_UnitContradictCheck(fact1, fact2, model)
        if label:
            print("Current Case is Contain Contradict")
            print("Fact1: ", fact1)
            print("Fact2: ", fact2)
            return 3
    counter = 0
    for fact1, fact2, status, nli_score in fact_pairs_list_blocked_nli:
        counter += 1
        if counter > 5 or nli_score < 0.02:
            break
        label = openai_UnitContradictCheck(fact1, fact2, model)
        if label:
            print("Current Case is Block Contradict")
            print("Fact1: ", fact1)
            print("Fact2: ", fact2)
            return 4
    return 5 # Not contradict

if __name__ == "__main__":
    # TODO: add the test code
    pass