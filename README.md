# User-Provided Relevance Feedback Information Retrieval System
COMS-E6111 Project 1


## How to run
- $`pip install -r requirements.txt`
- $`python3 main.py <google api key> <google engine id> <precision> <query>`
  -  Make sure the query terms are wrapped in parentheses


## Description
### Overall Structure
- `main(client_key, engine_key, query, precision)`
  - Entry point to the application, handling the google query iteration and collecting user input on google results
- `query_expansion(corpus, rel_idx, query)`
  - driver function for query expansion using rocchio's algorithm, handling vectorization of documents and running bigram reordering of query terms
- `rocchio(init_q_vec, doc_vecs, rel_idx)`
  - Implementation of Roccio's algorithm
- `reorder_query(related_docs, query, new_words)`
  - using bigram probability to reorder query terms 
### Details
- Reordering of query terms is implemented using ideas adapted from https://web.stanford.edu/~jurafsky/slp3/3.pdf. We used bigram probability to determine the best ordering of query terms. This is achieved by preprocessing the documents marked as related and removing stopwords and irrelevant characters. Then we leveraged `nltk` to generate bigrams. While going through the document, we count up the occurrence of each word and each bigram which we can use to calculate P(word| previous_word). If there is no occurence of such bigram, we check the rest of the words and find if there exist a bigram we can generate with the remaining words in all the query terms (old + new query terms)


## Imports
- `pprint`: print search results neatly
- `numpy`: vector space manipulation and other calculations
- `googleapiclient.discovery.build`: query Google
- `sklearn.feature_extraction.text.TfidfVectorizer`: calculate tf-idf weights, generate document vectors
- `sklearn.feature_extraction.text.ENGLISH_STOP_WORDS`: implement stop words
- `sklearn.metrics.pairwise.linear_kernel`: calculate cosine similarity
- `nltk`: used for stopword processing and bigram generation


## About
Authors: 
- Aryana Mohammadi (am5723)
- Jasmine Shin (yx2810)

Files included:
- `main.py`
- `README.md`
- `requirements.txt`
- `transcript.txt`

Keys:
- Google Custom Search Engine JSON API Key: AIzaSyDJZCCaWHDkdmn1dENt6Pynp9mcykVHtpg
- Engine ID: c6f4622f0b9a14652

Notes: 
- HTML and non-HTML filres are taken into consideration for every search and precision calculation.


