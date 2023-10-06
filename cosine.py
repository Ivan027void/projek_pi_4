import os
import glob
import heapq
import nltk
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from pre_tfidf import vectorizer
import time

# Create stopword remover object
factory = StopWordRemoverFactory()
stopword = factory.create_stop_word_remover()

# Create a stemming object
stemmer_factory = StemmerFactory()
stemmer = stemmer_factory.create_stemmer()

# Load the TF-IDF matrix
tfidf_matrix = np.load('tfidf_matrix.npy', allow_pickle=True)

# Tokenisasi, penghapusan stopwords, dan stemming pada query
query = input("Masukkan kata yang ingin dicari: ")
query = stopword.remove(query)  # Menghapus stopwords dari query
query_tokens = nltk.word_tokenize(query)  # Tokenisasi query
query_tokens = [stemmer.stem(word) for word in query_tokens]  # Melakukan stemming pada query

# Transform the query into a TF-IDF vector using the same vectorizer
query_vector = vectorizer.transform([' '.join(query_tokens)])

# Calculate cosine similarity between the query and documents
cosine_scores = cosine_similarity(query_vector, tfidf_matrix)

# Load document names from the text file
with open('document_names.txt', 'r', encoding='utf-8') as f:
    document_names = [line.strip() for line in f.readlines()]

# Create a list to store the results with document titles, scores, and positions of query words
results = []

scorpus_pi_path = 'C:/Users/ahini/Downloads/projek_pi_4/scorpus_pi/'

# Loop through the cosine scores and check if each document contains words from the query
for idx, score in enumerate(cosine_scores[0]):
    if score > 0:
        # If the cosine score is greater than 0, it means the document contains words from the query
        document_title = document_names[idx]  # Get the real document title
        positions = []

        # Open the document and find all the positions where the query word appears
        with open(os.path.join(scorpus_pi_path, document_title), 'r', encoding='utf-8') as f:
            document = f.read()

            for match in nltk.re.finditer(query, document):
                positions.append(match.start())

        results.append((score, document_title, positions))

# Sort the results by cosine score in descending order
results.sort(reverse=True)
start_time = time.time()
# Print the ranking with document titles, scores, and positions of query words
print("Dokumen teratas berdasarkan cosine similarity:")
for rank, (score, document_title, positions) in enumerate(results, start=1):
    print(f"Rank: {rank}")
    print(f"Nama Dokumen: {document_title}")
    print(f"Skor Cosine Similarity: {score:.8f}")
    print(f"Posisi Index untuk Kata dalam Query: {positions}")
    print()
    

end_time = time.time()

print(f"Waktu eksekusi: {end_time - start_time:.2f} detik")

