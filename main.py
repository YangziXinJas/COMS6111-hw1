import sys
import pprint
import numpy as np
import re
from googleapiclient.discovery import build
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import linear_kernel

from nltk.lm.preprocessing import pad_both_ends
from nltk.util import bigrams
from nltk.corpus import stopwords
import nltk

DEVELOPER_KEY = "AIzaSyDJZCCaWHDkdmn1dENt6Pynp9mcykVHtpg"
CX = "c6f4622f0b9a14652"
stop_words = ""

nltk.download('stopwords')

def main(client_key, engine_key, query, precision):
    '''Build a service object for interacting with the API. Visit
    the Google APIs Console <http://code.google.com/apis/console>
    to get an API key for your own application.'''
    
    # searc with google API
    result = search(query, client_key, engine_key)
    
    # terminate for <10 results
    query_size = len(result)
    if query_size < 10:
        print(f"Your search only returned {query_size} matches.")
        exit()
    
    relevance = -1
    search_num = 1
    while relevance / 10 < float(precision) or relevance == 0:
        print("\nparameters")
        print(f"Client Key = {client_key}")
        print(f"Engine Key = {engine_key}")
        print(f"Query      = {query}")
        print(f"Precision = {precision}")
        print(f"SEARCH #{search_num}")

        print("Google Search Results:")
        print("======================")

        relevance = 0
        corpus = [] 
        rel_idx = [] # indices of relevant results
        
        
        for i in range(0, len(result)):
            item = result[i]
            
            d = {}
            d['title'] = item['title']
            d['url'] = item['link']
            d['description'] = item['snippet']

            print(f"Result #{i+1}:")
            print(f"URL: {d['url']}")
            print(f"Title: {d['title']}")
            print(f"Summary: {d['description']}" )

            corpus.append(d['description'])
            if input("Relevant (Y/N)? ").capitalize() == 'Y':
                relevance += 1
                rel_idx.append(i)
                
            print("\n")

        if len(rel_idx) == 0:
            print("NO RELEVANT RESULTS FOUND, EXITING")
            exit()
                
        # output results
        print("=======================")
        print("FEEDBACK SUMMARY")
        print(f"Query: {query}")
        print(f"Precision: {relevance/10}")
            
        if relevance/10 < float(precision):
            print(f"Still below the desired precision of {precision}")
            query = query_expansion(corpus, rel_idx, query)
            result = search(query, client_key, engine_key) 
        
        search_num += 1



def get_stop_words():
    '''Generate array of stop words from given file'''
    
    global stop_words
    
    if len(stop_words) > 0:
        return
        
    file = "stop_words.txt"
    with open(file, "r") as file:
        stop_words = set(word.strip() for word in file)        

    return

def rocchio(init_q_vec, doc_vecs, rel_idx):
    '''Rocchio algorithm: q_opt = argmax[sim(q,C_r) - sim(q,C_nr)]
    rel: vector space of tf-idf values of relevant documents
    non-rel: vector space of tf-idf values of non-relevant documents'''
    
    # separate relevant document vectors
    rel = doc_vecs[rel_idx, :]
    non_rel = np.delete(doc_vecs, rel_idx, axis=0)
    
    # calculate centroids
    centroid_rel = np.mean(rel, axis=0)
    centroid_non = np.mean(non_rel, axis=0)
    
    # algorithm weights
    alpha = 1
    beta = 0.75
    gamma = 0.15
    
    # calculate modified query with Rocchio's
    mod_query_vec = (alpha * init_q_vec) + (beta * centroid_rel) - (gamma * centroid_non)
    
    return np.ravel(mod_query_vec)



def euclidian_length(vec):
    '''Returns the Euclidean Length of the 
    tf-idf vector representation of a document'''
    
    el = np.square(vec)
    el = np.sum(el)
    el = np.sqrt(el)
    
    return el
    
    
    
def similarity(d1, d2):
    '''Calculates Cosine Similarity of two documents, 
    given their tf-idf vectors'''
    
    numerator = np.dot(d1, d2)
    denominator = euclidian_length(d1) * euclidian_length(d2)
    
    try:
        sim = numerator / denominator
    except:
        sim = 0
        
    return sim
    
    
    
def query_expansion(corpus, rel_idx, query):
    '''Returns (string) new expanded query using tf-idf weights'''
    
    # obtain tf-idf vector representations
    vectorizer = TfidfVectorizer(stop_words=list(ENGLISH_STOP_WORDS))
    doc_vecs = vectorizer.fit_transform(corpus)
    doc_vecs = doc_vecs.toarray()
    init_q_vec = vectorizer.transform([query])
    feature_names = vectorizer.get_feature_names_out()
    
    # generate best words to add to the query
    mod_query_vec = rocchio(init_q_vec, doc_vecs, rel_idx)
    best_feature_idx = np.argsort(mod_query_vec)
    new_words = []
    i = -1
    while len(new_words) < 2 and i > -len(best_feature_idx):
        try_word = feature_names[best_feature_idx[i]]
        if not try_word in query.split():
            new_words.append(try_word)
        
        i -= 1

    print(f"Augmenting by words: {new_words}")
    new_query = reorder_query(np.take(corpus, rel_idx), query.split(), new_words)
    return new_query

def reorder_query(related_docs, query, new_words):
    bi_gram_count = {}
    word_count = {}
    word_count["<s>"] = len(related_docs)

    for doc in related_docs:
        # clean up punctuations and spaces
        doc = doc.lower()
        doc = re.sub('[^A-Za-z0-9 ]+', '', doc)
        doc = " ".join(doc.split())
        

        # clean up stopwords
        filtered_doc = [word for word in doc.split(" ") if word not in stopwords.words('english')]
        padded_bigrams = list(pad_both_ends(filtered_doc, n=2))

        for word in doc.split():
            if word in word_count:
                word_count[word] = word_count[word] + 1
            else:
                word_count[word] = 1

        # find all bigram occurences
        for bgram in list(bigrams(padded_bigrams)):
            if bgram in bi_gram_count:
                bi_gram_count[bgram] = bi_gram_count[bgram] + 1
            else:
                bi_gram_count[bgram] = 1

    query.extend(new_words)


    prev_word = "<s>"
    new_query = []
    while prev_word != query[-1]:
        Cx = word_count.get(prev_word, 0)
        max_Cxy = None
        max_P = -1
        for word in query:
            if word == prev_word or word in new_query:
                continue
            Cxy = bi_gram_count.get((prev_word, word), 0)
            probability = Cxy / Cx
            if probability > max_P:
                max_Cxy = (prev_word, word)
                max_P = probability
        new_query.append(max_Cxy[1])
        prev_word = max_Cxy[1]

    return " ".join(new_query)
    


def search(query, developer_key, cx):
    '''Returns (list) the results of a Google search (dict)
    of the given query (string)'''
    
    service = build(
        "customsearch", "v1", developerKey=developer_key
    )

    res = (
        service.cse()
        .list(
            q=query,
            cx=cx,
        )
        .execute()
    )
    
    try:
        result = res['items']
    except:
        result = []
        
    return result
    
    
    
if __name__ == "__main__":
    client_key = str(sys.argv[1])
    engine_key = str(sys.argv[2])
    precision = float(sys.argv[3])
    query = str(sys.argv[4])
    main(client_key, engine_key, query, precision)