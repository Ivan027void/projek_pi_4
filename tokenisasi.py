import os
import glob
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import nltk

# Inisialisasi Stopword Remover dan Stemmer
stopword_factory = StopWordRemoverFactory()
stemmer_factory = StemmerFactory()
stopword = stopword_factory.create_stop_word_remover()
stemmer = stemmer_factory.create_stemmer()

# Direktori dokumen
dir = 'C:/Users/ahini/Downloads/projek_pi_4/scorpus_pi/'
documents = []

# Get all txt files in the directory
files = glob.glob(os.path.join(dir, '*.txt'))

# Fungsi untuk melakukan preprocessing pada dokumen


def preprocess_document(document_text):
    # Hapus stopwords
    clean_document = stopword.remove(document_text)

    # Stemming kata-kata
    stemmed_document = []
    for word in clean_document.split():
        stemmed_word = stemmer.stem(word)
        stemmed_document.append(stemmed_word)

    # Tokenisasi dokumen
    tokenized_document = nltk.word_tokenize(' '.join(stemmed_document))

    return tokenized_document

# Loop through each file and read its content, kemudian preprocess
tokenized_clean_documents = []
for file in files:
    with open(file, 'r', encoding='utf-8') as f:
        text = f.read()
        tokenized_clean_document = preprocess_document(text)
        tokenized_clean_documents.append(tokenized_clean_document)
        print(
            f"Dokumen {len(tokenized_clean_documents)} berhasil di-token, dibersihkan, dan dipreprocessed")

# Create a new file to store the tokenized documents.
with open('tokenized_documents.txt', 'w', encoding='utf-8') as f:
    # For each document in the tokenized_clean_documents variable, write the document to the file, one document per line.
    for document in tokenized_clean_documents:
        # Write the document to the file, using a space as the delimiter.
        f.write(' '.join(document) + '\n')

# Close the file.
f.close()


def index_tokenized_words_in_document(tokenized_documents, output_file_path):
    # Create a dictionary to store the word index.
    word_index_dict = {}

    # Iterate over the tokenized documents.
    for document_index, tokenized_document in enumerate(tokenized_documents):

        # Iterate over the tokenized words in the document.
        for word_index, word in enumerate(tokenized_document):

            # If the word is not in the word index dictionary, add it.
            if word not in word_index_dict:
                word_index_dict[word] = []

            # Add the document index and word index to the word index dictionary.
            word_index_dict[word].append((document_index+1, word_index))

        # Sort the word index list for the current word.
        word_index_list = word_index_dict[word]
        word_index_list.sort()

    # Save the word index to a file.
    with open(output_file_path, 'w', encoding='utf-8') as f:
        for word, document_index_and_word_index_list in word_index_dict.items():
            f.write(f'{word}: {document_index_and_word_index_list}\n')


# Index all tokenized words in the documents and save the index to a file.
index_tokenized_words_in_document(tokenized_clean_documents, 'word_index_in_document_sorted.txt')