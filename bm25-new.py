import os
import glob
import heapq
import bisect
import re
import nltk
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
import numpy as np
from rank_bm25 import BM25Okapi

dir = 'C:/Users/ahini/Downloads/projek_pi_4/scorpus_pi/'
documents = []

# Get all txt files in the directory
files = glob.glob(os.path.join(dir, '*.txt'))

# Loop through each file and read its content
for file in files:
    with open(file, 'r', encoding='utf-8') as f:
        text = f.read()
        documents.append(text)


# Create stopword remover object
factory = StopWordRemoverFactory()
stopword = factory.create_stop_word_remover()

# Loop through each document and remove stopword
clean_documents = []
for document in documents:
    clean_document = stopword.remove(document)
    clean_documents.append(clean_document)


# Tokenize each document into words
tokenized_documents = [nltk.word_tokenize(
    document) for document in clean_documents]

# Create BM25 object
bm25 = BM25Okapi(tokenized_documents)

# Get user input query
query = input("Masukkan kata yang ingin dicari: ")

# Tokenize query into words
tokenized_query = nltk.word_tokenize(query)

# Get BM25 scores for each document
scores = bm25.get_scores(tokenized_query)

# Get the number of documents to display
n = int(input("Masukkan jumlah dokumen yang ingin ditampilkan: "))

# Get the indices of the highest scoring documents
indices = heapq.nlargest(n, range(len(scores)), scores.__getitem__)

# Print the ranked documents and the word indices
print(f"Dokumen teratas untuk query '{query}' adalah:")
for i in indices:
    print(f"Dokumen {i+1}:")
    print(f"Skor BM25: {scores[i]}")
    # Find the positions of the query words in the document
    positions = {}
    for word in tokenized_query:
        # Use binary search to find the index of the word
        index = bisect.bisect_left(tokenized_documents[i], word)
        # Check if the word is in the document
        if index < len(tokenized_documents[i]) and tokenized_documents[i][index] == word:
            # Add the word and its position to a dictionary
            positions[word] = positions.get(word, []) + [index]
    # Sort the positions by value
    positions = {k: sorted(v) for k, v in positions.items()}
    # Print the positions
    print(f"Posisi kata: {positions}")
    # Add asterisks around the query words in the document
    highlighted_document = re.sub(
        r'\b(' + '|'.join(tokenized_query) + r')\b', r'*\1*', documents[i])
    # Print the document
    print(f"Dokumen: {highlighted_document}")
