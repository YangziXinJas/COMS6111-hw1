import pprint
import numpy as np
from googleapiclient.discovery import build
from sklearn.feature_extraction.text import TfidfVectorizer

DEVELOPER_KEY = "AIzaSyDJZCCaWHDkdmn1dENt6Pynp9mcykVHtpg"
CX = "c6f4622f0b9a14652"
stop_words = ""


def main():
    '''Build a service object for interacting with the API. Visit
    the Google APIs Console <http://code.google.com/apis/console>
    to get an API key for your own application.'''
    
    # get_stop_words()
    
    # retrieve user information
    user_query = input("Enter search words: ")
    result = search(user_query)
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
        corpus = [] # snippets of relevant results
        
        for item in result:
            d = {}
            d['title'] = item['title']
            d['url'] = item['link']
            d['description'] = item['snippet']

            pprint.pprint(d)
            if input("Relevant (Y/N)? ").capitalize() == 'Y':
                relevance += 1
                corpus.append(d['description'])
            print("\n\n")
                
        print("\n\n\n\n\n")
        pprint.pprint(relevance/10)
            
        if relevance/10 < float(precision):
            query = query_expansion(corpus)
            # result = search(query)



def get_stop_words(d1, d2):
    '''Generate array of stop words from given file'''
    
    global stop_words
    
    if len(stop_words) > 0:
        return
        
    file = "stop_words.txt"
    with open(file, "r") as file:
        stop_words = set(word.strip() for word in file)        

    return


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
    
    
    
def query_expansion(corpus):
    '''Returns (string) new expanded query using tf-idf weights'''
    
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(corpus)
    feature_names = vectorizer.get_feature_names_out()
    X = X.toarray()
    pass



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