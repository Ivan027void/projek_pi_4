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

dir = 'C:/Users/ahini/Downloads/projek_pi_4/scorpus_pi/'
documents = []

# Get all txt files in the directory
files = glob.glob(os.path.join(dir, '*.txt'))

# Loop through each file and read its content
for file in files:
    with open(file, 'r', encoding='utf-8') as f:
        text = f.read()
        documents.append(text)

# Get user input query
query = input("Masukkan kata yang ingin dicari: ")

# Create stopword remover object
factory = StopWordRemoverFactory()
stopword = factory.create_stop_word_remover()

# Loop through each document and remove stopword
clean_documents = []
for document in documents:
    clean_document = stopword.remove(document)
    clean_documents.append(clean_document)

vectorizer = TfidfVectorizer(use_idf=True, smooth_idf=True)

tfidf_matrix = vectorizer.fit_transform(clean_documents)
features = vectorizer.get_feature_names_out()

query_vector = vectorizer.transform([query])

cosine_scores = cosine_similarity(query_vector, tfidf_matrix)

tokenized_query = nltk.word_tokenize(query)
tokenized_documents = [nltk.word_tokenize(
    document) for document in clean_documents]

bm25 = BM25Okapi(tokenized_documents)
bm25_scores = bm25.get_scores(tokenized_query)

def find_word_positions(features, tfidf_matrix):
    # Create an empty dictionary to store the word positions
    word_positions = {}
    # Loop through each feature (word)
    for i, word in enumerate(features):
        # Loop through each document (row) in the tfidf matrix
        for j, row in enumerate(tfidf_matrix):
            # Check if the word has a non-zero tf-idf value in the document
            if row[i] > 0:
                # Add the word and its position to the dictionary
                word_positions[word] = word_positions.get(word, []) + [j]
    # Return the dictionary of word positions
    return word_positions

# Call the function and store the result in a variable
word_positions = find_word_positions(features, tfidf_matrix.toarray())

n = int(input("Masukkan jumlah dokumen yang ingin ditampilkan: "))


# Get the indices of the highest scoring documents based on cosine similarity
cosine_indices = heapq.nlargest(n, range(len(cosine_scores[0])), cosine_scores[0].__getitem__)

# Get the indices of the highest scoring documents based on BM25 scores
bm25_indices = heapq.nlargest(n, range(len(bm25_scores)), bm25_scores.__getitem__)

for i in cosine_indices:
    print(f"Dokumen {i+1}:")
    print(f"Skor cosine similarity: {cosine_scores[0][i]}")
    
    # Find the positions of the query words in the document
    positions = {}
    for word in tokenized_query:
        # Check if the word is in the word positions dictionary
        if word in word_positions:
            # Check if the document index is in the word positions list for the current word
            if i in word_positions[word]:
                # Add the word and its positions to a dictionary
                positions[word] = word_positions[word][i]
    
    # Sort the positions by value
    positions = {k: sorted(v) for k, v in positions.items()}
    
    # Print the positions
    print(f"Posisi kata: {positions}")
    
    # Add asterisks around the query words in the document
    highlighted_document = documents[i]
    for word, positions_list in positions.items():
        for position in positions_list:
            highlighted_document = highlighted_document[:position] + '*' + word + '*' + highlighted_document[position+len(word):]
    
    # Print the document
    print(f"Dokumen: {highlighted_document}")

# Print a blank line to separate the results
print()

# Print the ranked documents and the word positions based on BM25 scores
print(f"Dokumen teratas untuk query '{query}' berdasarkan BM25 scores adalah:")
for i in bm25_indices:
    print(f"Dokumen {i+1}:")
    print(f"Skor BM25: {bm25_scores[i]}")
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