import string
import numpy as np
from rank_bm25 import BM25Okapi
import nltk
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
import os

# nltk.download('punkt')

# Inisialisasi StopWordRemover dari Sastrawi
stopword_factory = StopWordRemoverFactory()
stopword_remover = stopword_factory.create_stop_word_remover()

# Fungsi untuk membersihkan teks
def preprocess_text(text):
    # Menghapus tanda baca
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Mengubah teks menjadi huruf kecil
    text = text.lower()
    # Menghapus stop words menggunakan Sastrawi
    text = stopword_remover.remove(text)
    return text

# Fungsi untuk membaca semua dokumen dalam direktori
def read_documents(directory):
    documents = []
    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            with open(os.path.join(directory, filename), 'r', encoding='utf-8') as file:
                document_text = file.read()
                documents.append(document_text)
    return documents

# Fungsi untuk menghitung skor BM25
def calculate_bm25(documents):
    # Melakukan preprocessing dan tokenisasi pada dokumen
    tokenized_corpus = [nltk.word_tokenize(preprocess_text(document)) for document in documents]
    # Membuat objek BM25 dengan corpus yang telah ditokenisasi
    bm25 = BM25Okapi(tokenized_corpus)
    return bm25

# Fungsi untuk mencari indeks kata dalam dokumen
def find_word_index(word, documents):
    # Membuat sebuah list kosong untuk menyimpan hasil pencarian
    word_index = []
    # Menggunakan nltk untuk melakukan tokenisasi kata pada setiap dokumen
    for i, document in enumerate(documents):
        tokens = nltk.word_tokenize(document)
        # Mencari indeks kata dalam tokens dan menambahkannya ke list hasil pencarian
        for j, token in enumerate(tokens):
            if token == word:
                word_index.append((i, j))
    return word_index

# Fungsi untuk menampilkan hasil pencarian
def display_search_results(sorted_documents, word_index):
    # Menampilkan judul hasil pencarian
    print(f"Hasil pencarian untuk '{user_query}':")
    # Menampilkan dokumen yang diurutkan berdasarkan peringkat
    for rank, (index, score) in enumerate(sorted_documents, start=1):
        # Mendapatkan nama dokumen dari indeks
        document_name = os.listdir(dir)[index]
        # Mendapatkan indeks kata dari list hasil pencarian
        word_indices = [j for i, j in word_index if i == index]
        # Menampilkan peringkat, nama dokumen, dan indeks kata dalam dokumen
        print(f"Peringkat {rank}: {document_name}, Indeks Kata: {word_indices}")

# Membaca semua dokumen dalam direktori
dir = 'C:/Users/ahini/Downloads/projek_pi_4/scorpus_pi/'
documents = read_documents(dir)

# Menghitung skor BM25
bm25 = calculate_bm25(documents)

# Mengambil input dari pengguna
user_query = input("Masukkan term atau kata yang ingin Anda cari: ")

# Membersihkan dan menghitung skor BM25 untuk query pengguna
query_vector = nltk.word_tokenize(preprocess_text(user_query))
document_scores = bm25.get_scores(query_vector)

# Mendapatkan indeks dokumen yang diurutkan berdasarkan skor BM25 tertinggi
document_scores = [(i, score) for i, score in enumerate(document_scores)]
sorted_documents = sorted(document_scores, key=lambda x: x[1], reverse=True)

# Mencari indeks kata dalam dokumen
word_index = find_word_index(user_query, documents)

# Menampilkan hasil pencarian
display_search_results(sorted_documents, documents, word_index)

