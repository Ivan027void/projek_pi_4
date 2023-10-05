import os
import glob
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

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
stemmer = StemmerFactory().create_stemmer()

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

def index_tokenized_words_in_document(tokenized_documents, output_file_path):
  """Indexes all tokenized words in the given documents and saves the index to a file.

  Args:
    tokenized_documents: A list of lists of strings, where each sublist is a list of tokenized words in a document.
    output_file_path: The path to the output file.

  Returns:
    None.
  """
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