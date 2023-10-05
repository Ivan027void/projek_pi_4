import os
import glob
import heapq
import bisect
import re
import nltk
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# Create stopword remover object
factory = StopWordRemoverFactory()
stopword = factory.create_stop_word_remover()

# Create a stemming object
stemmer = nltk.PorterStemmer()

import tokenisasi

clean_documents = tokenisasi.clean_documents

# Tokenisasi, penghapusan stopwords, dan stemming pada query
query = input("Masukkan kata yang ingin dicari: ")
query = stopword.remove(query)  # Menghapus stopwords dari query
query_tokens = nltk.word_tokenize(query)  # Tokenisasi query
query_tokens = [stemmer.stem(word) for word in query_tokens]  # Melakukan stemming pada query

# Create a TF-IDF vectorizer
vectorizer = TfidfVectorizer(use_idf=True, smooth_idf=True)

# Calculate the TF-IDF vectors for the documents
tfidf_matrix = vectorizer.fit_transform(clean_documents)
features = vectorizer.get_feature_names_out()

# Transform the query into a TF-IDF vector
query_vector = vectorizer.transform([' '.join(query_tokens)])

# Calculate the cosine similarity scores between the query vector and the document vectors
cosine_scores = cosine_similarity(query_vector, tfidf_matrix)

tokenized_query = query_tokens 
tokenized_documents = [nltk.word_tokenize(
    document) for document in clean_documents]

bm25 = BM25Okapi(tokenized_documents)
bm25_scores = bm25.get_scores(tokenized_query)


# Membuat daftar dokumen yang relevan dengan kata-kata dalam query
relevant_documents = []
for i, document_tokens in enumerate(tokenized_documents):
    if any(word in document_tokens for word in query_tokens):
        relevant_documents.append(i)

# Get the top-ranked documents based on cosine similarity
cosine_indices = heapq.nlargest(len(relevant_documents), relevant_documents, key=lambda i: cosine_scores[0][i])

# Get the top-ranked documents based on BM25 scores
bm25_indices = heapq.nlargest(len(relevant_documents), relevant_documents, key=lambda i: bm25_scores[i])

# Get the document names
document_names = [os.path.basename(file) for file in tokenisasi.files]

# Print the ranking, document name, score, and word positions in the document
print("Dokumen teratas berdasarkan cosine similarity:")
for rank, i in enumerate(cosine_indices, start=1):
    print(f"Rank: {rank}")
    print(f"Nama Dokumen: {document_names[i]}")
    print(f"Skor Cosine Similarity: {cosine_scores[0][i]}")
    positions = [idx for idx, word in enumerate(tokenized_documents[i]) if word in query_tokens]
    print(f"Posisi Index untuk Kata dalam Query: {positions}")
    print()

print("Dokumen teratas berdasarkan BM25 scores:")
for rank, i in enumerate(bm25_indices, start=1):
    print(f"Rank: {rank}")
    print(f"Nama Dokumen: {document_names[i]}")
    print(f"Skor BM25: {bm25_scores[i]}")
    positions = [idx for idx, word in enumerate(tokenized_documents[i]) if word in query_tokens]
    print(f"Posisi Index untuk Kata dalam Query: {positions}")
    print()