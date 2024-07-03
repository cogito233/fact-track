import os
import pandas as pd
def subsample_byNLI(path, num_sample = 300):
    # path -> plot_contradict_maxNLI={numsamples}.csv
    df_fact = pd.read_csv(os.path.join(path, "fact.csv"))
    # plot_id	outline_id	fact_key	fact_content	l	r	fact_type
    dict_fact2plot = {}
    for i in range(len(df_fact)):
        row = df_fact.iloc[i]
        key = row["fact_key"]
        content = (row["outline_id"], row["plot_id"])
        if key not in dict_fact2plot:
            dict_fact2plot[key] = content
    df_fact_contradict = pd.read_csv(os.path.join(path, "fact_contradict.csv"))
    # fact1_key	fact2_key	nli_score	contradict	fact1_content	fact2_content
    # for each (outline_id, plot_id1, plot_id2), compute its max NLI score
    dict_maxNLI = {}
    for i in range(len(df_fact_contradict)):
        row = df_fact_contradict.iloc[i]
        plot_id1 = dict_fact2plot[row["fact1_key"]][1]
        plot_id2 = dict_fact2plot[row["fact2_key"]][1]
        nli_score = row["nli_score"]
        outline_id = dict_fact2plot[row["fact1_key"]][0]
        if key not in dict_maxNLI:
            dict_maxNLI[(outline_id, plot_id1, plot_id2)] = nli_score
        else:
            dict_maxNLI[(outline_id, plot_id1, plot_id2)] = max(dict_maxNLI[(outline_id, plot_id1, plot_id2)], nli_score)
    # sort by NLI score and select top num_sample
    sorted_maxNLI = sorted(dict_maxNLI.items(), key = lambda x: x[1], reverse = True)
    selected = sorted_maxNLI[:num_sample]
    selected = [x[0] for x in selected]

    df_contradict_temp = pd.read_csv(os.path.join(path, "plot_contradict_pred.csv"))
    # print(selected)
    # outline_id	plot_id1	plot_id2; we need add a column "type_byPred", by "contradict" or "unknown"
    for i in range(len(df_contradict_temp)):
        row = df_contradict_temp.iloc[i]
        outline_id = row["outline_id"]
        plot_id1 = row["plot1_id"]
        plot_id2 = row["plot2_id"]
        # print(outline_id, plot_id1, plot_id2)
        if outline_id == "1141_pure_simple" and plot_id1 == "1.2.3" and plot_id2 == "1.3":
            print(outline_id, plot_id1, plot_id2)
            print((outline_id, plot_id1, plot_id2) in selected)
        if (outline_id, plot_id1, plot_id2) in selected:
            df_contradict_temp.loc[i, "type_byPred"] = "contradict"
            print("contradict")
        else:
            df_contradict_temp.loc[i, "type_byPred"] = "unknown"
    outpath = os.path.join(path, f"plot_contradict_maxNLI={num_sample}.csv")
    print(df_contradict_temp)
    df_contradict_temp.to_csv(outpath)


if __name__ == "__main__":
    path = "/home/yangk/zhiheng/fact-track/dataPostAnno/1100_1200_pure_simple_llama2-7B_detect_gpt4_d3"
    subsample_byNLI(path, num_sample = 300)

