import math, os
from collections import Counter
import T0_ParsingFiles as parse

# Calculate doc ranking using BM25-based IR Model
def bm25(coll, word_freq, df):
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


# Calculate doc ranking using Jelinek-Mercer based Language Model
def jm_lm(coll, word_freq, df):
    scores = {}
    lambda_ = 0.4

    for docId, doc in coll.get_coll().items():
        termFreq = doc.get_term_list()
        scores[docId] = 1
        D = len(termFreq)
        Data_Cx = len(df)

        for word_i, _ in word_freq.items():
            f_qi_D = termFreq.get(word_i, 0)
            c_qi = df.get(word_i,0)

            val = ((1-lambda_)*f_qi_D/D) + (lambda_*c_qi/Data_Cx)
            if val != 0:
                scores[docId] *= val
    return scores


# Re-ranking documents using the query frequency.
def rerank_documents(coll, word_freq, df):
    scores = bm25(coll, word_freq, df)
    return scores

# Selecting top-k documents as pseudo-relevant for a query 
def select_top_k_documents(scores, k):
    sorted_docs = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    return [doc_id for doc_id, score in sorted_docs[:k]]

# Expanding the query using terms from the top-k documents
def expand_query(top_k_docs, collection, top_terms_count):
    term_counter = Counter()
    for doc_id in top_k_docs:
        terms = collection[doc_id].get_term_list()
        term_counter.update(terms)
    
    # Select top terms_count terms to add to the query
    significant_terms = dict()  #[term for term, count in term_counter.most_common(top_terms_count)]
    for term, count in term_counter.most_common(top_terms_count):
        significant_terms[term] = count
    return significant_terms

# Calcualte doc ranking using Pseudo-Relevance Model
def my_prm(coll, word_freq, df):
    # Number of top documents to consider as pseudo-relevant
    k = 15
    # Number of significant terms to select for query expansion
    top_terms_count = len(word_freq)

    # using BM25 algorithm for initial retrieval
    initial_scores = bm25(coll, word_freq, df)

    # Select top-k documents
    top_k_docs = select_top_k_documents(initial_scores, k)

    # Expand query
    expanded_terms = expand_query(top_k_docs, coll.get_coll(), top_terms_count)
        
    # Re-rank documents
    reranked_scores = rerank_documents(coll, expanded_terms, df)

    return reranked_scores


def print_save_score(docNum, query, bm25_scores, model):
    if "BM" in model:
        output_filepath = parse.output_folder + "/BM25_R"+ docNum + "Ranking.dat"
        t1_msg = f'\nThe query is: {query}\nThe following are the BM25 score for R{docNum} document:\n'
    elif "JM_LM" in model:
        output_filepath = parse.output_folder + "/JM_LM_R"+ docNum + "Ranking.dat"
        t1_msg = f'\nThe query is: {query}\nThe following are the JM_LM score for R{docNum} document:\n'
    else:
        output_filepath = parse.output_folder + "/MY_PRM_R"+ docNum + "Ranking.dat"
        t1_msg = f'\nThe query is: {query}\nThe following are the MY_PRM score for R{docNum} document:\n'

    file = open(output_filepath, "w")
    print(t1_msg, end = "")
    file.write(t1_msg)

    top_flag = 15
    for docId, score in dict(sorted(bm25_scores.items(), key=lambda x:x[1], reverse=True)).items():
        t1_msg = f'{docId} {score}\n'
        file.write(t1_msg)
        if top_flag > 0:
            print(t1_msg, end = "")
            top_flag -= 1

    return


# task 4 - implement all three model for each data collection with corresponding topic/query.
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
            # parsing documents for a collection
            collections = parse.parse_collection(coll_folderpath)

            # calculate avg doc len for this praticular data collection
            parse.avg_length(collections)
            # calculate document-frequency for given RcvlDoc collection
            df = parse.my_df(collections.get_coll())
            # parse the query
            word_freq = parse.parse_query(queries[coll_num])

            # calculate BM25-IR model score for respective data collection
            bm25_scores = bm25(collections, word_freq, df)
            print_save_score(coll_num, queries[coll_num], bm25_scores, "BM")

            # calculate Jelinek-Mercer model score for respective data collection
            jm_lm_scores = jm_lm(collections, word_freq, df)
            print_save_score(coll_num, queries[coll_num], jm_lm_scores, "JM_LM")

            # calculate Pseudo-Relevance model score for respective data collection
            my_prm_scores = my_prm(collections, word_freq, df)
            print_save_score(coll_num, queries[coll_num], my_prm_scores, "PRM")

    print("Completed!! the ranking scores are saved in RankingOutputs folder ")
    