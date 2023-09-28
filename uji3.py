import os
import glob
import heapq
import bisect
import re
import nltk
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import linear_kernel
from rank_bm25 import BM25Okapi

def read_documents(directory):
    documents = []
    stemmer = StemmerFactory().create_stemmer()
    stopword_remover = StopWordRemoverFactory().create_stop_word_remover()

    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            with open(os.path.join(directory, filename), 'r', encoding='utf-8') as file:
                document_text = file.read()

                # Remove all symbols and lowercase all text.
                document_text = re.sub(r'[^\w\s]', ' ', document_text)
                document_text = document_text.lower()

                # Stem the words in the document.
                document_text = ' '.join([stemmer.stem(word) for word in nltk.word_tokenize(document_text)])

                # Remove stopwords.
                document_text = ' '.join([word for word in nltk.word_tokenize(document_text) if word not in stopword_remover.remove(document_text)])

                documents.append(document_text)

    return documents

def calculate_tfidf(documents):
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(documents)
    return tfidf_vectorizer, tfidf_matrix

def calculate_cosine_similarity(query, documents, tfidf_matrix):
    query_vector = tfidf_vectorizer.transform([query])
    cosine_similarities = linear_kernel(query_vector, tfidf_matrix)
    return cosine_similarities

def calculate_bm25(documents, query):
    bm25_model = BM25Okapi(documents)
    bm25_scores = bm25_model.get_scores(query)
    return bm25_scores

def combine_results(tf_idf, cosine_similarity, bm25):
      # Make sure that all matrices have the same dimensions.
    if tf_idf.shape != cosine_similarity.shape or tf_idf.shape != bm25_scores.shape:
        raise ValueError("The input matrices must have same matrices")
    combined_results = tf_idf * cosine_similarity * bm25
    # Return the combined results.
    return combined_results

def display_results(documents, combined_results, directory, query):
    # Create a list of document names and combined scores.
    document_names = os.listdir(directory)
    combined_scores = list(zip(document_names, combined_results))

    # Sort the combined scores in descending order using heapq.
    heapq.heapify(combined_scores)
    combined_scores.sort(key=lambda x: x[1], reverse=True)

    # Print out the document ranking based on combined score with its name. Only rank the documents that have the word in the query.
    for rank, (document_name, score) in enumerate(combined_scores, start=1):
        if query in document_name:
            print(f"Peringkat {rank}: {document_name}, Skor: {score}")
        else:
            continue


if __name__ == '__main__':
    # Membaca semua dokumen
    dir = 'C:/Users/ahini/Downloads/projek_pi_4/scorpus_pi/'
    documents = read_documents(dir)
    
    # Menerima input query
    query = input("Masukkan query pencarian: ")

    # Menghitung TF-IDF
    tfidf_vectorizer, tfidf_matrix = calculate_tfidf(documents)

    # Menghitung cosine similarity
    cosine_similarities = calculate_cosine_similarity(query, documents, tfidf_matrix)

    # Menghitung BM25
    bm25_scores = calculate_bm25(documents, query)

    # Menggabungkan hasil TF-IDF, cosine similarity, dan BM25
    combined_results = combine_results(tfidf_matrix, cosine_similarities, bm25_scores)

    # Menampilkan hasil pencarian
    display_results(documents, combined_results, dir)