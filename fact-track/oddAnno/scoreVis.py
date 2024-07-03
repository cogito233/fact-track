import pandas as pd


def visDis(result_path):
    # Load the CSV file
    df = pd.read_csv(result_path)

    # Calculate the distribution of score_1 and score_2
    score_1_distribution = df['score_1'].value_counts().sort_index().to_dict()
    score_2_distribution = df['score_2'].value_counts().sort_index().to_dict()

    # Print or return the distributions
    print("Score 1 Distribution:", score_1_distribution)
    print("Score 2 Distribution:", score_2_distribution)

if __name__ == "__main__":
    path = "/cluster/project/sachan/zhiheng/zhiheng/fact-track/dataPostAnno/1100_1200_pure_simple_detect_llama2-7B/oddExp/scoreAnno_simple_random_llamaPairwise&llamaDect.csv"
    visDis(path)