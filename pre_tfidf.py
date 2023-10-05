import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# Create a TF-IDF vectorizer
vectorizer = TfidfVectorizer(use_idf=True, smooth_idf=True)

# Open the file for reading
with open('tokenized_documents.txt', 'r', encoding='utf-8') as f:
    # Read the contents of the file
    documents = [line.strip() for line in f]

# Close the file
f.close()

# Calculate the TF-IDF vectors for the documents
tfidf_matrix = vectorizer.fit_transform(documents)

# Save the TF-IDF vectors to a file
with open('tfidf_matrix.npy', 'wb') as f:
    np.save(f, tfidf_matrix)