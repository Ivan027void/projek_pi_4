import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
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

# Fungsi untuk menghitung skor TF-IDF
def calculate_tfidf(documents):
    tfidf_vectorizer = TfidfVectorizer(preprocessor=preprocess_text)
    tfidf_matrix = tfidf_vectorizer.fit_transform(documents)
    return tfidf_vectorizer, tfidf_matrix

# Fungsi untuk mencari indeks kata dalam dokumen
def find_word_index(word, documents, tfidf_vectorizer):
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

def filter_documents(documents, word_indices):
    filtered_documents = []
    for i, j in word_indices:
        filtered_documents.append(documents[i])
    return filtered_documents

def check_word_existence(document, word):
    # Menggunakan nltk untuk melakukan tokenisasi kata pada dokumen
    tokens = nltk.word_tokenize(document)
    # Memeriksa apakah kata ada dalam dokumen
    return word in tokens

def display_search_results(sorted_documents,documents, word_index):
    # Menampilkan judul hasil pencarian
    print(f"Hasil pencarian untuk '{user_query}':")
    # Menampilkan dokumen yang diurutkan berdasarkan peringkat
    for rank, (index, score) in enumerate(sorted_documents, start=1):
        # Mendapatkan nama dokumen dari indeks
        document_name = os.listdir(dir)[index]
        # Mendapatkan indeks kata dari list hasil pencarian
        word_indices = [j for i, j in word_index if i == index]
        # Menampilkan peringkat, nama dokumen, dan indeks kata dalam dokumen
        if check_word_existence(documents[index], user_query):
            print(f"Peringkat {rank}: {document_name}")
            print(f"score cosine similarity:{score} ,Indeks Kata: {word_indices}")
        
# Membaca semua dokumen dalam direktori
dir = 'C:/Users/ahini/Downloads/projek_pi_4/scorpus_pi/'
documents = read_documents(dir)

# Menghitung skor TF-IDF
tfidf_vectorizer, tfidf_matrix = calculate_tfidf(documents)

# Mengambil input dari pengguna
user_query = input("Masukkan term atau kata yang ingin Anda cari: ")

# Membersihkan dan menghitung TF-IDF untuk query pengguna
query_vector = tfidf_vectorizer.transform([preprocess_text(user_query)])

# Menghitung kesamaan kosinus antara query dan dokumen
cosine_similarities = linear_kernel(query_vector, tfidf_matrix).flatten()

# Mendapatkan indeks dokumen yang diurutkan berdasarkan kesamaan kosinus tertinggi
document_scores = [(i, score) for i, score in enumerate(cosine_similarities)]
sorted_documents = sorted(document_scores, key=lambda x: x[1], reverse=True)

# Mencari indeks kata dalam dokumen
word_index = find_word_index(user_query, documents, tfidf_vectorizer)

# Menampilkan hasil pencarian
display_search_results(sorted_documents, documents, word_index)
