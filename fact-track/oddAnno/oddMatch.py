import pandas as pd
import os
BASE_PATH = os.environ["BASE_PATH"]

import random

def oddMatch_crossOutline(result_dir, suffix = "gpt4Anno&llamaDect.csv", strategy = "max_nli", seed=42):
    pass

def oddMatch(result_dir, suffix = "gpt4Anno&llamaDect.csv", strategy = "max_nli", seed=42):
    """
    Function to match and sample 'fp' and 'fn' type data points from badcase_df.
    Each outline_id will have randomly sampled data points, with the number of samples
    being the minimum of the counts of 'fp' and 'fn' types.

    Args:
    result_dir (str): Directory path where the files are stored.
    suffix (str): Suffix for the filenames.
    strategy (str): Strategy for matching, default is 'max_nli', other option is 'random'
    seed (int): Seed for random number generator to ensure reproducibility.

    Returns:
    None: The function will save the sampled data into result_path.
    """
    # Setting the random seed for reproducibility
    random.seed(seed)


    # Paths for badcase and result files
    badcase_path = f"{result_dir}/badcase_{suffix}"
    result_path = f"{result_dir}/oddExp/oddMatch_{strategy}_{suffix}"

    if os.path.exists(result_path):
        print(f"{result_path} already exists.")
        return

    # Reading the badcase data
    badcase_df = pd.read_csv(badcase_path)

    # Initializing a list to store the results
    results = []

    # Grouping the dataframe by 'outline_id' and iterating through each group
    for outline_id, group in badcase_df.groupby('outline_id'):
        # Filtering 'fp' and 'fn' types
        fp_df = group[group['type'] == 'fp']
        fn_df = group[group['type'] == 'fn']

        # Determining the number of samples to take (minimum of counts of 'fp' and 'fn')
        n_samples = min(len(fp_df), len(fn_df))

        # Sampling data points based on the strategy
        if strategy == "max_nli":
            # Selecting the top n_samples by max_nli
            sampled_fp = fp_df.nlargest(n_samples, 'max_nli').reset_index(drop=True)
            sampled_fn = fn_df.nlargest(n_samples, 'max_nli').reset_index(drop=True)
        elif strategy == "random":
            # Selecting a random sample of size n_samples
            sampled_fp = fp_df.sample(n_samples).reset_index(drop=True)
            sampled_fn = fn_df.sample(n_samples).reset_index(drop=True)
        else:
            raise ValueError("Invalid strategy. Valid options are 'max_nli' and 'random'.")

        # Combining the sampled points
        for i in range(n_samples):
            result = {
                'outline_id': outline_id,
                'plot_1A': sampled_fp.at[i, 'plot_1'],
                'plot_1B': sampled_fp.at[i, 'plot_2'],
                'type_1': 'fp',
                'max_nli_1': sampled_fp.at[i, 'max_nli'],
                'plot_2A': sampled_fn.at[i, 'plot_1'],
                'plot_2B': sampled_fn.at[i, 'plot_2'],
                'type_2': 'fn',
                'max_nli_2': sampled_fn.at[i, 'max_nli']
            }
            # print("#"*20)
            # print(result['plot_1A'])
            # print(result['plot_1B'])
            # print("$"*20)
            # print(result['plot_2A'])
            # print(result['plot_2B'])
            # print("#"*20)

            results.append(result)
    # exit(0)
    # Converting the results list to a DataFrame
    result_df = pd.DataFrame(results)

    # Saving the result DataFrame to a CSV file
    result_df.to_csv(result_path, index=False)
    print(len(result_df))

    return result_df

# Note: The function won't execute here as it requires file access.
# You can use this function in your environment where the data files are accessible.


if __name__ == "__main__":
    # 假设我们已经有了一个fp和fn的list，我们需要在同一个outline里进行match

    # result_dir = f"{BASE_PATH}/fact-track/dataPostAnno/101x_pure_simple_llama2-7B_block=0.8"

    result_dir = f"{BASE_PATH}/fact-track/dataPostAnno/1100_1200_pure_simple_detect_llama2-7B"
    oddMatch(result_dir, strategy="max_nli")
    oddMatch(result_dir, strategy="random")