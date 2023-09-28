import os
import glob
import heapq
import bisect
import re
import nltk
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory


class Document:
    def __init__(self, text):
        self.text = text
        self.tokens = nltk.word_tokenize(text)
        self.factory = StopWordRemoverFactory()
        self.stopwords = self.factory.create_stop_word_remover()
        self.stemmer = nltk.PorterStemmer()
        self.clean_text = ' '.join(
            [self.stemmer.stem(word) for word in self.stopwords.remove(text)])
        self.vectorizer = TfidfVectorizer(use_idf=True, smooth_idf=True)
        self.vectorizer.fit([self.clean_text])
        self.tfidf_vector = self.vectorizer.transform([self.clean_text])
        self.bm25 = BM25Okapi(self.text)
        self.bm25_score = self.bm25.get_scores([self.tokens])

    def get_cosine_similarity_score(self, query_vector):
        return cosine_similarity(self.tfidf_vector, query_vector)[0][0]


class Query:
    def __init__(self, text):
        self.text = text
        self.tokens = nltk.word_tokenize(text)

    def get_tfidf_vector(self):
        return self.vectorizer.transform([self.text])[0]

    def find_word_positions(features, tfidf_matrix):
        """Menemukan posisi kata dalam dokumen.

        Args:
            features: Fitur-fitur dokumen.
            tfidf_matrix: Matriks TF-IDF dokumen.

        Returns:
            Kamus posisi kata dalam dokumen.
        """

        word_positions = {}
        for i, word in enumerate(features):
            for j, row in enumerate(tfidf_matrix):
                if row[i] > 0:
                    word_positions[word] = word_positions.get(word, []) + [j]
        return word_positions


def rank_documents(documents, query, method='cosine_similarity', doc_num=10):
    """Meranking dokumen berdasarkan skor kemiripan.

    Args:
        documents: Daftar dokumen.
        query: Kueri pencarian.
        method: Metode perankingan ('cosine_similarity' atau 'bm25').
        doc_num: Jumlah dokumen yang akan ditampilkan.

    Returns:
        Daftar indeks dokumen yang telah diurutkan berdasarkan skor kemiripan.
    """

    if method == 'cosine_similarity':
        query_vector = query.get_tfidf_vector()
        scores = [document.get_cosine_similarity_score(
            query_vector) for document in documents]
        return heapq.nlargest(doc_num, range(len(scores)), scores.__getitem__)
    elif method == 'bm25':
        scores = [document.bm25_score[0] for document in documents]
        return heapq.nlargest(doc_num, range(len(scores)), scores.__getitem__)
    else:
        raise ValueError('Invalid ranking method')


def highlight_query_words(document, query):
    """Menyorot kata kueri dalam dokumen.

    Args:
        document: Dokumen.
        query: Kueri pencarian.

    Returns:
        Dokumen dengan kata kueri yang disorot.
    """

    highlighted_document = re.sub(
        r'\b(' + '|'.join(query.tokens) + r')\b', r'*\1*', document.text)
    return highlighted_document


def main():
   # Get all txt files in the directory
    dir = 'C:/Users/ahini/Downloads/projek_pi_4/scorpus_pi/'
    files = glob.glob(os.path.join(dir, '*.txt'))

    # Create a list of documents
    documents = []
    for file in files:
        with open(file, 'r', encoding='utf-8') as f:
            text = f.read()
            documents.append(Document(text))

    # Get user input query
    query = input("Masukkan kata yang ingin dicari: ")

    # Get user input for the number of documents to return
    n = int(input("Masukkan jumlah dokumen yang ingin ditampilkan: "))

    # Create a query object
    query = Query(query)

    # Rank the documents
    ranked_documents = rank_documents(documents, query, n)

    # Print the ranked documents
    print(f"Dokumen teratas untuk query '{query.text}' adalah:")
    for i in ranked_documents:
        print(f"Dokumen {i+1}:")
        print(f"Skor: {ranked_documents[i]}")
        print(f"Dokumen: {highlight_query_words(documents[i], query)}")


if __name__ == '__main__':
    main()
