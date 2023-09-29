import os
import glob
import nltk
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

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

# Create a stemming object
stemmer = nltk.PorterStemmer()

# Loop through each document and remove stopwords and stem the words
clean_documents = []
for document in documents:
    clean_document = stopword.remove(document)
    stemmed_document = []
    for word in clean_document.split():
        stemmed_word = stemmer.stem(word)
        stemmed_document.append(stemmed_word)
    clean_documents.append(' '.join(stemmed_document))

# Tokenisasi dan stemming semua kata dalam dokumen yang bersih
tokenized_clean_documents = []
for document in clean_documents:
    tokenized_clean_document = []
    for word in document.split():
        stemmed_word = stemmer.stem(word)
        tokenized_clean_document.append(stemmed_word)
    tokenized_clean_documents.append(tokenized_clean_document)

# Open the output file
with open('all_tokenized_words.txt', 'w', encoding='utf-8') as f:
    # Loop through each document
    for tokenized_document in tokenized_clean_documents:
        # Write the tokenized words of the document to the output file
        f.write(' '.join(tokenized_document) + '\n')
        
