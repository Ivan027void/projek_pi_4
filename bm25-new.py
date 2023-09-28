import os
import glob
import re
import nltk
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# Lokasi direktori dokumen
dir = 'C:/Users/ahini/Downloads/projek_pi_4/scorpus_pi/'
documents = []

# Get all txt files in the directory
files = glob.glob(os.path.join(dir, '*.txt'))

# Loop through each file and read its content
for file in files:
    with open(file, 'r', encoding='utf-8') as f:
        text = f.read()
        documents.append(text)

# Input query dari pengguna
query = input("Masukkan kata yang ingin dicari: ")

# Create a stopword remover object
factory = StopWordRemoverFactory()
stopword = factory.create_stop_word_remover()

# Create a stemming object
stemmer = nltk.PorterStemmer()

# Tokenize the query
tokenized_query = nltk.word_tokenize(query)

# Remove stop words and stem the query words
stemmed_query = []
for word in tokenized_query:
    if word not in stopword:
        stemmed_query.append(stemmer.stem(word))

# Create a TF-IDF vectorizer
vectorizer = TfidfVectorizer(use_idf=True, smooth_idf=True)

# Calculate the TF-IDF vectors for the documents
tfidf_matrix = vectorizer.fit_transform(documents)
features = vectorizer.get_feature_names_out()

# Create a query vector
query_vector = vectorizer.transform([query])

# Calculate the cosine similarity scores
cosine_scores = cosine_similarity(query_vector, tfidf_matrix).flatten()

# Sort the documents in descending order by cosine similarity score
ranked_documents = np.argsort(cosine_scores)[::-1]

# Find the positions of the query words in the documents
word_positions = {}
for word in stemmed_query:
    word_positions[word] = []
    for i, document in enumerate(documents):
        if word in document:
            word_positions[word].append(i)

# Print the ranked documents and the word positions
print(f"Dokumen teratas untuk query '{query}' berdasarkan cosine similarity adalah:")
for i in ranked_documents[:10]:
    print(f"Dokumen {i+1}:")
    print(f"Skor cosine similarity: {cosine_scores[0][i]}")

    # Find the positions of the query words in the document
    positions = {}
    for word in stemmed_query:
        if word in word_positions:
            if i in word_positions[word]:
                positions[word] = positions.get(word, []) + [i]

    # Sort the positions by value
    positions = {k: sorted(v) for k, v in positions.items()}

    # Print the positions
    print(f"Posisi kata: {positions}")

    # Add asterisks around the query words in the document
    highlighted_document = re.sub(
        r'\b(' + '|'.join(stemmed_query) + r')\b', r'*\1*', documents[i])

    # Print the document
    print(f"Dokumen: {highlighted_document}")
    print()
