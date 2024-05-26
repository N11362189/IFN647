import numpy as np
from scipy.stats import ttest_rel, ttest_ind
import pandas as pd

def perform_t_tests(bm25_results, jm_lm_results, my_prm_results):
    measures = ['AP']
    comparisons = [
        ('BM25', 'JM_LM', bm25_results, jm_lm_results),
        ('BM25', 'My_PRM', bm25_results, my_prm_results),
        ('JM_LM', 'My_PRM', jm_lm_results, my_prm_results)
    ]
    
    t_test_results = {}
    
    for measure in measures:
        t_test_results[measure] = {}
        for model1, model2, res1, res2 in comparisons:
            t_stat, p_value = ttest_rel(res1[measure], res2[measure])
            t_test_results[measure][f'{model1} vs {model2}'] = (t_stat, p_value)
    
    return t_test_results



if __name__ == "__main__":
    avg_prc = pd.read_csv("avg_prc_df.csv", header=[0])
    print(avg_prc)

    bm25_data = {}
    jm_lm_data = {}
    my_prm_data = {}

    bm25_data["AP"] = pd.to_numeric(avg_prc["BM25"]).to_list()
    jm_lm_data["AP"] = pd.to_numeric(avg_prc["JM_LM"]).to_list()
    my_prm_data["AP"] = pd.to_numeric(avg_prc["My_PRM"]).to_list()

    # print(bm25_data["AP"])

    t_test_results = perform_t_tests(bm25_data, jm_lm_data, my_prm_data)

    for measure, results in t_test_results.items():
        print(f"\nT-test results for {measure}:")
        for comparison, (t_stat, p_value) in results.items():
            print(f"{comparison}: t-statistic = {t_stat:.4f}, p-value = {p_value:.4f}")