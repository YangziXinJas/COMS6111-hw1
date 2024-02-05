import pprint
import numpy as np
from googleapiclient.discovery import build
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

DEVELOPER_KEY = "AIzaSyDJZCCaWHDkdmn1dENt6Pynp9mcykVHtpg"
CX = "c6f4622f0b9a14652"
stop_words = ""


def main():
    '''Build a service object for interacting with the API. Visit
    the Google APIs Console <http://code.google.com/apis/console>
    to get an API key for your own application.'''
    
    # get_stop_words()
    
    # retrieve user information
    query = input("Enter search words: ")
    result = search(query)
    precision = input("Enter desired precision (between 0 and 1): ")
    while True:
        try:
            precision = float(precision)
            assert precision >= 0 or precision <= 1
            break
        except:
            precision = input("Invalid. Please enter a value between 0 and 1: ")
            pass
    print("\n\n")
    
    # terminate for <10 results
    query_size = len(result)
    if query_size < 10:
        print(f"Your search only returned {query_size} matches.")
        exit()

    
    relevance = -1
    while relevance / 10 < float(precision) or relevance == 0:
        relevance = 0
        corpus = [] 
        rel_idx = [] # indices of relevant results
        
        
        for i in range(0, len(result)):
            item = result[i]
            
            d = {}
            d['title'] = item['title']
            d['url'] = item['link']
            d['description'] = item['snippet']

            pprint.pprint(d)
            corpus.append(d['description'])
            if input("Relevant (Y/N)? ").capitalize() == 'Y':
                relevance += 1
                rel_idx.append(i)
                
            print("\n\n")
                
        print("\n\n\n\n\n")
        pprint.pprint(relevance/10)
            
        if relevance/10 < float(precision):
            query = query_expansion(corpus, rel_idx, query)
            result = search(query) 



def get_stop_words(d1, d2):
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
    vectorizer = TfidfVectorizer()
    doc_vecs = vectorizer.fit_transform(corpus)
    doc_vecs = doc_vecs.toarray()
    init_q_vec = vectorizer.transform([query])
    feature_names = vectorizer.get_feature_names_out()
    
    mod_query_vec = rocchio(init_q_vec, doc_vecs, rel_idx)
    # generate best words to add to the query
    best_feature_idx = np.argsort(mod_query_vec)
    
    new_words = []
    i = -1
    while len(new_words) < 2 and i > -len(best_feature_idx):
        try_word = feature_names[best_feature_idx[i]]
        if not try_word in query.split():
            new_words.append(try_word)
        
        i -= 1
    
    new_query = query + " " + " ".join(new_words)
    
    print(f"New Query: {new_query}")
    
    # exit()
    return new_query



def search(query):
    '''Returns (list) the results of a Google search (dict)
    of the given query (string)'''
    
    service = build(
        "customsearch", "v1", developerKey=DEVELOPER_KEY
    )

    res = (
        service.cse()
        .list(
            q=query,
            cx=CX,
        )
        .execute()
    )
    
    try:
        result = res['items']
    except:
        result = []
        
    return result
    
    
    
if __name__ == "__main__":
    main()