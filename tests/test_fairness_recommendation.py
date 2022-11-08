import pandas as pd
from src.FairPkg.fairness_recommendation import fairness_tree_metric


def test_fairness_tree_metric():
    ftree_df = pd.read_csv("src/FairPkg/fairness_data/fairness_tree.csv")
    #results = fairness_tree_metric(ftree_df)
    #print(results)
