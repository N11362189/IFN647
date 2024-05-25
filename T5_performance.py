import T0_ParsingFiles as parse
import pandas as pd

# calculate the avg precision of a single collection
def avg_precision(bnk, ranks):
    ri = 0
    map1 = 0.0
    # R = len([id for (id,v) in bnk.items() if v>0])
    for (n,id) in sorted(ranks.items(), key=lambda x: int(x[0])):
        if (bnk[id]>0):
            ri =ri+1
            pi = float(ri)/float(int(n))
            map1 = map1 + pi
            # recall = float(ri)/float(R)
            # print("At position " + str(int(n)) + ", precision= " + str(pi) + ", recall= " + str(recall))
    return map1/float(ri)

# calculate the avg precision for all 3 models of a single collection
def coll_avg_prc(bnk, bm25, jm_lm, prm):
    return avg_precision(bnk, bm25), avg_precision(bnk, jm_lm), avg_precision(bnk, prm)

# calculating the performance of 3 models on average precision(MAP)
def compare_avg_precision(colls_bnks):
    response = []

    # calculating average precision for each collection
    for collId, bnk in colls_bnks.items():
        bm25 = parse.read_output_file('BM25_R' + collId + 'Ranking.dat')
        jm_lm = parse.read_output_file('JM_LM_R' + collId + 'Ranking.dat')
        prm = parse.read_output_file('MY_PRM_R' + collId + 'Ranking.dat')
        
        bm25_prc, jmlm_prc, prm_prc = coll_avg_prc(bnk, bm25, jm_lm, prm)
        response.append({'Topic':'R'+collId, 
                    'BM25':bm25_prc,
                    'JM_LM':jmlm_prc,
                    'My_PRM':prm_prc})
    
    # create table to store the precision values for each topic/collection
    avg_prc_df = pd.DataFrame(response).sort_values(by='Topic')
    # calculating the average precision (MAP) for the data collection
    avg_df = pd.DataFrame([{'Topic':'MAP', 
                           'BM25': avg_prc_df['BM25'].mean(),
                            'JM_LM': avg_prc_df['JM_LM'].mean(),
                            'My_PRM': avg_prc_df['My_PRM'].mean()}])
    avg_prc_df = pd.concat([avg_prc_df, avg_df], ignore_index=True)
    return avg_prc_df


# calculating the performance of 3 models on precision@10
def compare_precision10(colls_bnks):
    response = []
    
    # create table to store the precision values for each topic/collection
    prc10_df = pd.DataFrame(response).sort_values(by='Topic')
    # calculating the average precision (MAP) for the data collection
    avg_df = pd.DataFrame([{'Topic':'Average', 
                           'BM25': avg_prc_df['BM25'].mean(),
                            'JM_LM': avg_prc_df['JM_LM'].mean(),
                            'My_PRM': avg_prc_df['My_PRM'].mean()}])
    prc10_df = pd.concat([prc10_df, avg_df], ignore_index=True)
    return prc10_df


# calculating the performance of 3 models on DCG10
def compare_dcg10(colls_bnks):
    response = []
    
    # create table to store the precision values for each topic/collection
    dcg10_df = pd.DataFrame(response).sort_values(by='Topic')
    # calculating the average precision (MAP) for the data collection
    avg_df = pd.DataFrame([{'Topic':'Average', 
                           'BM25': avg_prc_df['BM25'].mean(),
                            'JM_LM': avg_prc_df['JM_LM'].mean(),
                            'My_PRM': avg_prc_df['My_PRM'].mean()}])
    dcg10_df = pd.concat([dcg10_df, avg_df], ignore_index=True)
    return dcg10_df


# task 5 - implement three different effectiveness measures to evaluate the document ranking
if __name__ == "__main__":
    # reading evaluation benmarks folder
    colls_benchmarks = parse.evaluation_benchmark()

    # calculating performance of 3 models on average precision(MAP)
    avg_prc_df = compare_avg_precision(colls_benchmarks)
    print("\n The performance of 3 models on average precision (MAP)\n")
    print(avg_prc_df)

    # calculating performance of 3 models on precision@10
    # prc10_df = compare_precision10(colls_benchmarks)
    # print("\n The performance of 3 models on precision@10\n")
    # print(prc10_df)

    # calculating performance of 3 models on DCG10
    # dcg10_df = compare_dcg10(colls_benchmarks)
    # print("\n The performance of 3 models on DCG10\n")
    # print(dcg10_df)