import pandas as pd
import numpy as np
import sys
import os
BASE_PATH = os.environ["BASE_PATH"]

def inference_factPair(path):
    # Input: all plot and fact information;
    # Output: for all plot pair, find the corresponding class;
    plotDF = pd.read_csv(path + "/plot.csv") # plot_id	outline_id	plot_content
    # plotDF.set_index(['plot_id', 'outline_id'], inplace=True)
    factDF = pd.read_csv(path + "/fact.csv") # plot_id	outline_id	fact_key	fact_content	l	r	fact_type
    factDF.set_index("fact_key", inplace=True)
    # Build a set of plot and fact
    sys.path.append(f"{BASE_PATH}/fact-track/fact-track/utils")
    from huggingface_api import get_embedding_contriever, similarity_from_embedding, huggingface_UnitContradictScore
    # Step 1: 找到所有fact pair，一前一后，并且similarity>0.5; 并且计算contradict性质，overlap性质
    # factPair_overlap.csv: (outline_id, plot_id1, fact_id1, plot_id2, fact_id2, similarity, contradict, overlap)
    # Step 1.1: fact_id -> embedding
    if os.path.exists(path + "/factEmbedding.npy"):
        factEmbedding_dict = np.load(path + "/factEmbedding.npy", allow_pickle=True).item()
    else:
        factEmbedding_dict = {}
        from tqdm import tqdm
        for fact_id in tqdm(factDF.index):
            fact = factDF.loc[fact_id, "fact_content"]   # Access using .loc and the index
            if type(fact) != str:
                fact = fact.iloc[0]
            # print("#"*100)
            # print(fact)
            # print("#"*100)
            factEmbedding_dict[fact_id] = get_embedding_contriever(fact)
        np.save(path + "/factEmbedding.npy", factEmbedding_dict)
    print("Finish embedding")
    # Step 1.2: for all fact pair, calculate similarity
    # first iterate all outline_id
    outline_id_list = plotDF["outline_id"].unique()
    with open(path + "/factPair_overlap.csv", "w") as f:
        f.write("outline_id,plot_id1,fact_id1,plot_id2,fact_id2,similarity,contradict,overlap\n")
    from tqdm import tqdm
    # Open the file once for writing
    with open(path + "/factPair_overlap.csv", "a") as f:
        for outline_id in tqdm(outline_id_list):
            factDF_sub = factDF[factDF["outline_id"] == outline_id]
            fact_id_list = factDF_sub.index.unique()
            # print(outline_id)
            # print(fact_id_list)
            # Iterate over fact_id list
            for i in range(len(fact_id_list)):
                current_entry = factDF_sub.loc[fact_id_list[i]]
                if isinstance(current_entry, pd.DataFrame):
                    fact_i = current_entry.iloc[0]
                else:
                    fact_i = current_entry
                # Access the first occurrence if duplicates
                # print(fact_i)
                # print(factDF_sub.loc[fact_id_list[i]])
                # print(type(factDF_sub.loc[fact_id_list[i]]))
                # print(fact_i["fact_type"])
                if fact_i["fact_type"] == "postfact":
                    for j in range(len(fact_id_list)):
                        current_entry = factDF_sub.loc[fact_id_list[j]]
                        if isinstance(current_entry, pd.DataFrame):
                            fact_j = current_entry.iloc[0]
                        else:
                            fact_j = current_entry
                        # Access the first occurrence if duplicates
                        if fact_j["fact_type"] == "prefact":
                            plot_id1, plot_id2 = fact_i["plot_id"], fact_j["plot_id"]
                            # print(plot_id1, plot_id2, fact_i["fact_content"], fact_j["fact_content"])
                            if plot_id1[:min(len(plot_id1), len(plot_id2))] != plot_id2[
                                                                               :min(len(plot_id1), len(plot_id2))]:
                                similarity = similarity_from_embedding(factEmbedding_dict[fact_id_list[i]],
                                                                       factEmbedding_dict[fact_id_list[j]])
                                if similarity >= 0.5:
                                    contradict = huggingface_UnitContradictScore(fact_i["fact_content"],
                                                                                 fact_j["fact_content"])
                                    l1, r1, l2, r2 = fact_i["l"], fact_i["r"], fact_j["l"], fact_j["r"]
                                    if l2 < l1 and l1 < r2 and r2 < r1:
                                        overlap = 2  # Full Overlap
                                    elif l1 < r1 and r1 < l2 and l2 < r2:
                                        overlap = 0  # Not Overlap
                                    else:
                                        overlap = 1  # Partial Overlap

                                    # Write to file
                                    f.write(
                                        f"{outline_id},{plot_id1},{fact_id_list[i]},{plot_id2},{fact_id_list[j]},{similarity},{contradict},{overlap}\n")
import csv

import os
import csv
import pandas as pd


def postProcess_factPair(path, inputFile="factPair_overlap2.csv", outputFile="factPair_overlap2_new.csv"):
    input_file = os.path.join(path, inputFile)
    output_file = os.path.join(path, outputFile)

    with open(input_file, 'r', newline='') as infile, open(output_file, 'w', newline='') as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile)
        flag = False
        for row in reader:
            fixed_row = []
            for i in range(len(row)):
                # If it's the third (index 2) or fifth (index 4) expected column
                if len(fixed_row) in [2, 4]:
                    # Check if the item is not a hex string
                    if not all(c in '0123456789abcdef' for c in row[i]) and flag:
                        # Append with a comma, and ensure entire field is enclosed in quotes
                        fixed_row[-1] = f'{fixed_row[-1]},{row[i]}'
                    else:
                        fixed_row.append(row[i])
                else:
                    fixed_row.append(row[i])
            if len(fixed_row) == 0:
                continue
            # Write the row to the output file
            writer.writerow(fixed_row)
            flag = True

def inference_plotPair(path):
    # Load DataFrames
    plot_temp_df = pd.read_csv(f"{path}/plot_contradict_temp.csv")
    factPair_overlap_df = pd.read_csv(f"{path}/factPair_overlap_new.csv")
    # print(factPair_overlap_df.head())
    # exit(0)

    # Merge DataFrames on (outline_id, plot_id1, plot_id2) <=> (outline_id, plot1_id, plot2_id)
    merged_df = pd.merge(plot_temp_df, factPair_overlap_df, left_on=["plot1_id", "plot2_id", "outline_id"],
                         right_on=["plot_id1", "plot_id2", "outline_id"])

    # Group by plot1_id, plot2_id, outline_id, and overlap, then find max contradict score for each group
    grouped = merged_df.groupby(['plot1_id', 'plot2_id', 'outline_id', 'overlap']).agg(
        {'contradict': 'max'}).reset_index()

    # Pivot table to spread overlap categories into separate columns with custom names
    pivot_df = grouped.pivot_table(index=['plot1_id', 'plot2_id', 'outline_id'], columns='overlap', values='contradict')

    # Rename columns to include the overlap level in the name
    pivot_df.columns = [f'contradict_{int(col)}' for col in pivot_df.columns]

    # Reset index to turn multi-index into columns
    result_df = pivot_df.reset_index()

    # Fill NaN values with -1 where contradict scores are missing
    result_df.fillna(-1, inplace=True)

    # Save to CSV (optional)
    result_df.to_csv(f"{path}/plot_contradict_overlap.csv", index=False)
    print(result_df)

    return result_df


def subsample(path, threshold=0.2358, sample_size=500):
    # Load initial data
    df_plotOverlap = pd.read_csv(os.path.join(path, "plot_contradict_overlap.csv"))

    # Initialize a mask to track rows that are still available for sampling
    available_mask = pd.Series(True, index=df_plotOverlap.index)

    # Process four iterations for subsampling
    for i in range(0, 4):
        # Reset 'type_byPred' for all rows at the beginning of each iteration
        df_plotOverlap['type_byPred'] = 'unknown'
        if (i!=3):
            # Filter data where contradict_2 is above the threshold and the row is still available
            filtered = df_plotOverlap[(df_plotOverlap[f"contradict_{2-i}"] > threshold) & available_mask]
        else:
            filtered = df_plotOverlap[available_mask]

        print(len(filtered))
        # Check if there are enough rows to sample, otherwise take all available
        if len(filtered) > sample_size:
            sampled = filtered.sample(n=sample_size, random_state=i)  # Random state changes with i for variability
        else:
            sampled = filtered

        # Update available_mask to exclude sampled rows
        available_mask[filtered.index] = False

        # Assign 'contradict' to sampled rows
        df_plotOverlap.loc[sampled.index, 'type_byPred'] = 'contradict'

        # Save the results to a new CSV file for this iteration
        df_plotOverlap[['plot1_id', 'plot2_id', 'outline_id', 'type_byPred']].to_csv(
            os.path.join(path, f"plot_contradict_overlap{i}.csv"), index=False)


def subsample_union(path, threshold=0.2358, sample_size=500):
    # Load initial data
    df_plotOverlap = pd.read_csv(os.path.join(path, "plot_contradict_overlap.csv"))

    # Define the columns to check for the condition
    contradict_columns = ['contradict_0', 'contradict_1', 'contradict_2']

    # Create a mask where any of the contradict values exceeds the threshold
    union_mask = (df_plotOverlap[contradict_columns] > threshold).any(axis=1)

    # Filter the DataFrame based on the union_mask
    filtered = df_plotOverlap[union_mask]

    print("Filtered rows count:", len(filtered))

    # Check if there are enough rows to sample, otherwise take all available
    if len(filtered) > sample_size:
        sampled = filtered.sample(n=sample_size, random_state=42)  # Consistent random state for reproducibility
    else:
        sampled = filtered

    # Assign 'contradict' to sampled rows
    df_plotOverlap['type_byPred'] = 'unknown'  # Reset all to 'unknown'
    df_plotOverlap.loc[sampled.index, 'type_byPred'] = 'contradict'

    # Save the results to a new CSV file
    output_file = os.path.join(path, "plot_contradict_overlap_0-2.csv")
    print(f"Saving to {output_file}")
    df_plotOverlap[['plot1_id', 'plot2_id', 'outline_id', 'type_byPred']].to_csv(output_file, index=False)

    return output_file

def interval_abalation_exp(path):
    from scoreAnno_pos import scoreAnnotation
    input_name = "plot_contradict_overlap0.csv"
    output_name = "plot_scoreOutline_overlap0.csv"
    scoreAnnotation(path, input_name, output_name, is_simple=False)

    input_name = "plot_contradict_overlap_0-2.csv"
    output_name = "plot_scoreOutline_overlap_0-2.csv"
    scoreAnnotation(path, input_name, output_name, is_simple=False)

    from scoreAnno_pos import scoreAnnotation
    input_name = "plot_contradict_overlap0.csv"
    output_name = "plot_scoreSimple_overlap0.csv"
    scoreAnnotation(path, input_name, output_name, is_simple=False)

    input_name = "plot_contradict_overlap_0-2.csv"
    output_name = "plot_scoreSimple_overlap_0-2.csv"
    scoreAnnotation(path, input_name, output_name, is_simple=False)

def vis_ablation_exp(path):
    from scoreVis_pos import calculate_statistics
    calculate_statistics("plot_scoreOutline_overlap0.csv", path)
    calculate_statistics("plot_scoreOutline_overlap_0-2.csv", path)
    calculate_statistics("plot_scoreSimple_overlap0.csv", path)
    calculate_statistics("plot_scoreSimple_overlap_0-2.csv", path)

if __name__ == "__main__":
    # path = "/home/yangk/zhiheng/fact-track/dataPostAnno/1100_1200_pure_simple_detect_llama2-7B"
    path = "/home/yangk/zhiheng/fact-track/dataPostAnno/1100_1200_pure_simple_llama2-7B_detect_gpt4_d3"
    # inference_factPair(path) # Done
    # postProcess_factPair(path, inputFile="factPair_overlap.csv", outputFile="factPair_overlap_new.csv")
    # inference_plotPair(path)
    # subsample(path)
    # subsample_union(path)
    # interval_abalation_exp(path)
    vis_ablation_exp(path)