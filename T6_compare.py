import numpy as np
from scipy.stats import ttest_rel, ttest_ind
import pandas as pd

# evaluating t-test for different models
def t_test_eval(bm25_results, jm_lm_results, my_prm_results):
    #different measures
    measures = ['AP', 'DCG10', 'PRC10']
    #comparision of the models
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
    avg_prc = pd.read_csv("T5_AvgPrecision.csv", header=[0])
    dcg10 = pd.read_csv("T5_DCG10.csv", header=[0])
    precision10 = pd.read_csv("T5_Precision10.csv", header=[0])

    bm25_data = {}
    jm_lm_data = {}
    my_prm_data = {}
    
    measures = ['AP', 'DCG10', 'PRC10']
    measure_data = [avg_prc, dcg10, precision10]
    # creating dictionaries with measures data frm different csv files of task 5
    for i in range(0, len(measures)):
            bm25_data[measures[i]] = pd.to_numeric(measure_data[i]["BM25"]).to_list()
            jm_lm_data[measures[i]] = pd.to_numeric(measure_data[i]["JM_LM"]).to_list()
            my_prm_data[measures[i]] = pd.to_numeric(measure_data[i]["My_PRM"]).to_list()

    t_test_results = t_test_eval(bm25_data, jm_lm_data, my_prm_data)
    
    # print result of t-test
    for measure, results in t_test_results.items():
        print(f"\nT-test results for {measure}:")
        for comparison, (t_stat, p_value) in results.items():
            print(f"{comparison}: t-statistic = {t_stat:.4f}, p-value = {p_value:.4f}")