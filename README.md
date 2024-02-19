# User-Provided Relevance Feedback Information Retrieval System
COMS-E6111 Project 1


## How to run
- $`pip install -r requirements.txt`
- $`python3 main.py <google api key> <google engine id> <precision> <query>`
  -  Make sure the query terms are wrapped in parentheses


## Description
### Overall Structure
- main(client_key, engine_key, query, precision)
  - Entry point to the application, handling the google query iteration and collecting user input on google results
- query_expansion(corpus, rel_idx, query)
  - driver function for query expansion using rocchio's algorithm, handling vectorization of documents and running bigram reordering of query terms
- rocchio(init_q_vec, doc_vecs, rel_idx)
  - Implementation of Roccio's algorithm
- reorder_query(related_docs, query, new_words)
  - using bigram probability to reorder query terms 
### Details
- [A detailed description of your query-modification method (this is the core component of the project); this description should cover all important details of how you select the new keywords to add in each round, as well as of how you determine the query word order in each round]


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


