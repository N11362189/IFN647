import math, os
import T0_ParsingFiles as parse

# Calculate BM25 IR score for query q with respective coll
def bm25(coll, q, df):
    word_freq = parse.parse_query(q)
    scores = {}
    R, ri = 0, 0
    k1, k2, b = 1.2, 500, 0.75
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
            # base of the log function is 10
            scores[docId] += (math.log(log_val, 10) * k1_attr * k2_attr)

        if scores[docId] < 0:
            scores[docId] = 0.0

    return scores

# task 1 - BM25-based IR model for each data collection with corresponding topic/query.
if __name__ == "__main__":
    # parse 50 queries and save in dict() collId: {query frequency}
    queries = parse.parse_queryfile()
    # print(queries)

    folders = [folder for folder in os.listdir(parse.data_collection_folder)]
    for folder in folders:
        coll_folderpath = parse.data_collection_folder + "/" + folder
        # check if folder length is 9
        if len(folder) == 9:
            coll_num = folder[-3:]
            print(f"Calculating BM25-IR ranking scores for {folder} data collection")
            # parsing documents for a collection
            collections = parse.parse_collection(coll_folderpath)

            # calculate avg doc len for this praticular data collection
            parse.avg_length(collections)

            # calculate document-frequency for given RcvlDoc collection
            df = parse.my_df(collections.get_coll())

            # calculate BM25-IR model score for respective data collection
            bm25_scores = bm25(collections, queries[coll_num], df)

            # save the rankings for each doc in RankingOutputs folder
            output_filepath = parse.outputFolder + "/BM25_R" + coll_num + "Ranking.dat"
            file = open(output_filepath, "w")
            t1_msg = f'\nThe query is: {queries[coll_num]}\nThe following are the BM25 score for each document:\n\n'
            file.write(t1_msg)

            for docId, score in dict(sorted(bm25_scores.items(), key=lambda x:x[1], reverse=True)).items():
                t1_msg = f'{docId} {score}\n'
                file.write(t1_msg)

    print("Completed!! the ranking scores are saved in RankingOutputs folder ")
