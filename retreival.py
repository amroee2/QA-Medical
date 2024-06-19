import pandas as pd
from rank_bm25 import BM25Okapi
import nltk

# Load the CSV file
df = pd.read_csv('train.csv')

# Tokenize the documents
nltk.download('punkt')
tokenized_corpus = [nltk.word_tokenize(doc.lower()) for doc in df['question']]

# Initialize the BM25 model
bm25 = BM25Okapi(tokenized_corpus)

# Function to retrieve the most relevant documents
def retrieve_context(query, bm25, df, top_n=1):
    tokenized_query = nltk.word_tokenize(query.lower())
    scores = bm25.get_scores(tokenized_query)
    best_doc_indices = scores.argsort()[-top_n:][::-1]
    return df.iloc[best_doc_indices]['answer'].tolist()

# Example query
query = "اكلت الكيك ودخت هل انا مريض سكري"

# Retrieve the most relevant documents
relevant_docs = retrieve_context(query, bm25, df, top_n=1)
print(f"no of docs= {len(relevant_docs)}")
print(f"Relevant documents: {relevant_docs}")

