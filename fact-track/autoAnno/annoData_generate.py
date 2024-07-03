import pandas as pd
import os
BASE_PATH = os.environ["BASE_PATH"]

import sys
sys.path.append(f"{BASE_PATH}/fact-track/fact-track/analyze")

from convert_dataTemplate import metaNames2detectionData

def convert(begin, end, suffix = "pure_simple_llama2-7B", output = None):
    if output is None:
        output = f"{suffix}_{begin}_{end}.csv"
    metanames = []
    for i in range(begin, end):
        metaname = f"{i}_{suffix}"
        path = f"{BASE_PATH}/fact-track/data/{metaname}/object/outline.pkl"
        if os.path.exists(path):
            metanames.append(metaname)
    metaNames2detectionData(metanames, output)

if __name__ == "__main__":
    convert(1100, 1200)