import os, math
import string, re
from stemming.porter2 import stem

# fetch stoping words from the file
def get_stop_words(filepath):
    file = open(filepath)
    content = file.readline().split(',')
    return content

stop_words = get_stop_words("common-english-words.txt")
queryfile = "./the50Queries.txt"
output_folder = "./RankingOutputs"
benchmark_folder = "./EvaluationBenchmark"
data_collection_folder = "./Data_Collection"

class DataDoc:
    def __init__(self, docID):
        self.docID = docID
        self.terms = dict()
        self.doc_len = 0
        
    def set_doc_len(self, count):
        self.doc_len = count
        return
    
    def getDocId(self):
        return self.docID
 
    def getDocLen(self):
        return self.doc_len

    def get_term_list(self):
        return dict(sorted(self.terms.items(), key=lambda x:x[1], reverse=True))   
    
    def add_term(self, term):
        if term not in self.terms:
            self.terms[term] = 1
        else:
            self.terms[term] += 1
        return

class DataColl:
    def __init__(self):
        self.collections = {}
        self.numOfDocs = 0
        self.totalDocLength = 0
        self.avgDocLen = 0

    def add_doc(self, doc):
        self.collections[doc.getDocId()] = doc
        self.totalDocLength += doc.getDocLen()
        self.numOfDocs += 1
        return

    def get_coll(self):
        return self.collections
    
    def getTotalDocLen(self):
        return self.totalDocLength
    
    def getNumOfDocs(self):
        return self.numOfDocs
    
    def setAvgDocLen(self, avg):
        self.avgDocLen = avg

    def getAvgDocLen(self):
        return self.avgDocLen

# parsing a query
def parse_query(query):
    curr_word  = dict()

    query = query.translate(str.maketrans('','', string.digits)).translate(str.maketrans(string.punctuation, ' '*len(string.punctuation)))
    query = re.sub("\s+", " ", query)
    for term in query.split():
        term = term.lower()
        if len(term) > 2 and term not in stop_words:
            try:
                curr_word[term] += 1
            except KeyError:
                curr_word[term] = 1

    return curr_word

# parsing given query txt file
def parse_queryfile():
    i, queries  = 0, dict()
    file = open(queryfile).readlines()

    while i<len(file):
        if '<num>' in file[i]:
            numline = file[i].strip()
            while True:
                i += 1
                if '<title>' in file[i]:
                    title = file[i].replace('<title>', '').strip()
                    queries[numline[-3:]] = title
                    break
            i += 1
        else:
            i += 1

    return queries

# parsing documents in a collection
def parse_collection(inputpath):
    coll = DataColl()
    files = os.listdir(inputpath)
    for file in files:
        file_path = os.path.join(inputpath, file)
        start_end = False
        word_count = 0

        if os.path.isfile(file_path):
            myfile=open(file_path)
            file_=myfile.readlines()
            for line in file_:
                line = line.strip()
                if(start_end == False):
                    if line.startswith("<newsitem "):
                        for part in line.split():
                            if part.startswith("itemid="):
                                dataDoc = DataDoc(part.split("=")[1].split("\"")[1])
                                break
                    if line.startswith("<text>"):
                        start_end = True  
                elif line.startswith("</text>"):
                    break
                else:
                    line = line.replace("<p>", "").replace("</p>", "")
                    line = line.translate(str.maketrans('','', string.digits)).translate(str.maketrans(string.punctuation, ' '*len(string.punctuation)))
                    line = re.sub("\s+", " ", line)
                    for term in line.split():
                        word_count += 1
                        stemmed_term = stem(term.lower())
                        if len(stemmed_term) > 2 and stemmed_term not in stop_words:
                            try:
                                dataDoc.add_term(stemmed_term)
                            except KeyError:
                                dataDoc.add_term(stemmed_term)
            myfile.close()
            dataDoc.set_doc_len(word_count)
            coll.add_doc(dataDoc)

    return coll

# Calculate document-frequency for given data collection
def my_df(coll):
    docFreq = {}
    for _, doc in coll.items():
        terms = doc.get_term_list()
        for term in terms:
            docFreq[term] = docFreq.get(term, 0) + 1
    return docFreq

# Calculate and return avg length of all docs in a data coll 
def avg_length(coll):
    avg_length = coll.getTotalDocLen()/coll.getNumOfDocs()
    coll.setAvgDocLen(avg_length)
    return avg_length

# reading evaluation benchmark folder
def evaluation_benchmark():
    colls_bnk = dict()
    files = [folder for folder in os.listdir(benchmark_folder)]

    for file in files:
        coll_bnk = dict()
        content = open(benchmark_folder + "/" + file).readlines()
        for line in content:
            coll, docID, val = line.strip().split(" ")
            coll_bnk[docID] = float(val)
        colls_bnk[coll[1:]] = coll_bnk
        
    return colls_bnk

# reading ranking output files
def read_rankingOutputs():
    filenames = []
    files = [folder for folder in os.listdir(output_folder)]
    for file in files:
        filenames.append(output_folder + "/" + file)
    return filenames

def read_output_file(filename):
    ranks={}
    flag, i = 1, 1
    
    for line in open(output_folder + "/" + filename):
        if flag > 4:
            line = line.strip()
            line1 = line.split()
            ranks[str(i)] = line1[0]
            i = i + 1
        else:
            flag += 1
    return ranks