import math, os
import T0_ParsingFiles as parse

outputT1Folder = "../RankingOutputs"
data_collection_folder = "./Data_Collection"

# Calculate BM25 IR score for query q with respective coll
def my_bm25(coll, q, df):
    word_freq = parse.parse_query(q)
    scores = {}
    R, ri = 0, 0
    k1, k2, b = 1.2, 100, 0.9
    N = coll.getNumOfDocs()
    avg_docLen = coll.getAvgDocLen()

    for docId, doc in coll.get_coll().items():
        termFreq = doc.get_term_list()
        scores[docId] = 0
        dl_avdl = doc.getDocLen()/avg_docLen
        K = k1 * ((1-b) + (b*dl_avdl))

        for word_i, qfi in word_freq.items():
            ni = df.get(word_i,0)
            fi = termFreq.get(word_i, 0)

            log_val = ((ri+0.5)/(R-ri+0.5)) / ((ni-ri+0.5)/(N-ni-R+ri+0.5))
            k1_attr = ((k1+1)*fi) / (K+fi)    
            k2_attr  = ((k2+1)*qfi) / (k2+qfi)
            scores[docId] += (math.log(log_val, 2) * k1_attr * k2_attr)

    return scores

# task 1 - BM25-based IR model for each data collection with corresponding topic/query.
if __name__ == "__main__":
    # parse 50 queries and save in dict() collId: {query frequency}
    queries = parse.parse_queryfile()
    # print(queries)

    folders = [folder for folder in os.listdir(data_collection_folder)]
    for folder in folders:
        coll_folderpath = data_collection_folder + "/" + folder
        # check if folder length is 9
        if len(folder) == 9:
            if folder[-3:] == "150":
                # parsing documents
                collections = parse.parse_collection(coll_folderpath)
                collections.print_and_save()
                collDocs = collections.get_coll()
                # calculate document-frequency for given RcvlDoc collection
                df = parse.my_df(collDocs)