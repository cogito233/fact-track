path_default = "/home/yangk/zhiheng/fact-track/dataPostAnno/1100_1200_pure_simple_detect_llama2-7B"

import os
import pandas as pd
import numpy as np
from scipy.stats import norm


def calculate_statistics(filename, path = None):
    if path is None:
        path = path_default
    # Load the DataFrame
    df = pd.read_csv(os.path.join(path, filename))
    print(f"Loaded {filename}")
    # Determine the column to use
    column_to_use = 'score_simple' if 'score_simple' in df.columns else 'score_outline'

    # Filter non-NaN values from the selected column
    valid_scores = df[column_to_use].dropna()
    valid_scores = [int(i) for i in valid_scores if i in [1,2,3,4,5,'1','2','3','4','5']]
    valid_scores = pd.Series(valid_scores)
    # Calculate total count (N)
    N = valid_scores.count()
    print(f"Total count: {N}")
    # Calculate distribution
    distribution = valid_scores.value_counts().to_dict()
    print(f"Distribution: {distribution}")

    # Calculate average
    average = valid_scores.mean()

    # Calculate standard deviation
    std_dev = valid_scores.std()

    # Calculate the 95% confidence interval
    Z = norm.ppf(0.975)  # Z-score for 95% confidence
    margin_of_error = Z * (std_dev / np.sqrt(N))
    confidence_interval = (average - margin_of_error, average + margin_of_error)
    # keep 3 decimal places
    average = round(average, 3)
    confidence_interval = tuple(map(lambda x: round(x, 3), confidence_interval))
    print(f"Average: {average}, 95% Confidence Interval: {confidence_interval}")
    return {
        "N": N,
        "Distribution": distribution,
        "Average": average,
        "95% Confidence Interval": confidence_interval
    }

def vis_simpleResult():
    # output_name = "plot_scoreSimple_anno.csv"
    # calculate_statistics(output_name)
    # output_name = "plot_scoreSimple_baselineLlama.csv"
    # calculate_statistics(output_name)
    # output_name = "plot_scoreSimple_pred.csv"
    # calculate_statistics(output_name)
    # output_name = "plot_scoreSimple_random.csv"
    # calculate_statistics(output_name)
    # output_name = "plot_scoreSimple_maxNLI.csv"
    # calculate_statistics(output_name)

    output_name = "plot_scoreSimple_overlap0.csv"
    calculate_statistics(output_name)
    output_name = "plot_scoreSimple_overlap1.csv"
    calculate_statistics(output_name)
    output_name = "plot_scoreSimple_overlap2.csv"
    calculate_statistics(output_name)
    output_name = "plot_scoreSimple_overlap3.csv"
    calculate_statistics(output_name)

def vis_outlineResult():
    # output_name = "plot_scoreOutline_anno.csv"
    # calculate_statistics(output_name)
    # output_name = "plot_scoreOutline_baselineLlama.csv"
    # calculate_statistics(output_name)
    # output_name = "plot_scoreOutline_pred.csv"
    # calculate_statistics(output_name)
    # output_name = "plot_scoreOutline_random.csv"
    # calculate_statistics(output_name)
    # output_name = "plot_scoreOutline_maxNLI.csv"
    # calculate_statistics(output_name)
    # output_name = "plot_scoreOutline_overlap0.csv"
    # calculate_statistics(output_name)
    # output_name = "plot_scoreOutline_overlap1.csv"
    # calculate_statistics(output_name)
    # output_name = "plot_scoreOutline_overlap2.csv"
    # calculate_statistics(output_name)
    # output_name = "plot_scoreOutline_overlap3.csv"
    # calculate_statistics(output_name)
    output_name = "plot_scoreOutline_maxNLI=300_gpt4.csv"
    calculate_statistics(output_name)
    output_name = "plot_scoreSimple_maxNLI=300_gpt4.csv"
    calculate_statistics(output_name)

if __name__ == "__main__":
    vis_simpleResult()
    vis_outlineResult()