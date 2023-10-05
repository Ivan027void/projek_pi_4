import os
import glob
import heapq
import nltk
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from pre_tfidf import vectorizer

# Create stopword remover object
factory = StopWordRemoverFactory()
stopword = factory.create_stop_word_remover()

# Create a stemming object
stemmer_factory = StemmerFactory()
stemmer = stemmer_factory.create_stemmer()

# Load the TF-IDF vectors from the file
with open('tfidf_matrix.npy', 'rb') as f:
    tfidf_matrix = np.load(f)

# Load the document names
document_names = [os.path.basename(file) for file in glob.glob('tokenized_documents/*.txt')]

# Tokenisasi, penghapusan stopwords, dan stemming pada query
query = input("Masukkan kata yang ingin dicari: ")
query = stopword.remove(query)  # Menghapus stopwords dari query
query_tokens = nltk.word_tokenize(query)  # Tokenisasi query
query_tokens = [stemmer.stem(word) for word in query_tokens]  # Melakukan stemming pada query

# Transform the query into a TF-IDF vector
query_vector = vectorizer.transform([' '.join(query_tokens)])

# Calculate the cosine similarity scores between the query vector and the document vectors
cosine_scores = cosine_similarity(query_vector, tfidf_matrix)

# Retrieve the top-ranked documents based on cosine similarity scores
relevant_documents = []
for i, cosine_score in enumerate(cosine_scores[0]):
    if cosine_score > 0:
        relevant_documents.append(i)

# Get the top-ranked documents
cosine_indices = heapq.nlargest(len(relevant_documents), relevant_documents, key=lambda i: cosine_scores[0][i])

# Print the ranking, document name, score, and word positions in the document
print("Dokumen teratas berdasarkan cosine similarity:")
for rank, i in enumerate(cosine_indices, start=1):
    print(f"Rank: {rank}")
    print(f"Nama Dokumen: {document_names[i]}")
    print(f"Skor Cosine Similarity: {cosine_scores[0][i]}")
    positions = [idx for idx, word in enumerate(document_names[i].split()) if word in query_tokens]
    print(f"Posisi Index untuk Kata dalam Query: {positions}")
    print()
