import math
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import time

class BM25:
    def __init__(self, tokenized_documents, k1=1.5, b=0.75):
        self.documents = tokenized_documents
        self.doc_lengths = [len(doc) for doc in tokenized_documents]
        self.avg_doc_length = sum(self.doc_lengths) / len(tokenized_documents)
        self.k1 = k1  # Parameter k1 dalam BM25
        self.b = b    # Parameter b dalam BM25
        self.doc_term_freqs = []
        self.inverted_index = {}
        self.idf_values = {}
        self.doc_count = len(tokenized_documents)
        self._initialize()

    def _initialize(self):
        for doc_index, doc in enumerate(self.documents):
            doc_term_freq = {}
            for term in doc:
                if term not in doc_term_freq:
                    doc_term_freq[term] = 0
                doc_term_freq[term] += 1
                if term not in self.inverted_index:
                    self.inverted_index[term] = []
                if doc_index not in self.inverted_index[term]:
                    self.inverted_index[term].append(doc_index)
            self.doc_term_freqs.append(doc_term_freq)

        for term in self.inverted_index:
            doc_freq = len(self.inverted_index[term])
            self.idf_values[term] = math.log(
                (self.doc_count - doc_freq + 0.5) / (doc_freq + 0.5) + 1.0)

    def _calculate_score(self, query_terms, doc_index):
        score = 0.0
        doc_length = self.doc_lengths[doc_index]

        for term in query_terms:
            if term in self.inverted_index:
                df = len(self.inverted_index[term])
                idf = self.idf_values[term]
                tf = self.doc_term_freqs[doc_index].get(term, 0)
                numerator = (tf * (self.k1 + 1))
                denominator = (tf + self.k1 * (1 - self.b + self.b * (doc_length / self.avg_doc_length)))
                score += idf * (numerator / denominator)

        return score

    # def search(self, query, top_n=300):
    #     query_terms = query.split()
    #     scores = []

    #     for doc_index in range(self.doc_count):
    #         score = self._calculate_score(query_terms, doc_index)
    #         scores.append((doc_index, score))

    #     scores.sort(key=lambda x: x[1], reverse=True)
    #     return scores[:top_n]
    
    def search(self, query):
        query_terms = query.split()
        relevant_documents = []

        for doc_index in range(self.doc_count):
            score = self._calculate_score(query_terms, doc_index)
            if score > 0:
                relevant_documents.append((doc_index, score))

        relevant_documents.sort(key=lambda x: x[1], reverse=True)
        return relevant_documents


if __name__ == "__main__":
    # Load tokenized documents from file
    with open('tokenized_documents.txt', 'r', encoding='utf-8') as f:
        tokenized_documents = [line.strip().split() for line in f.readlines()]

    # Create a BM25 instance
    bm25 = BM25(tokenized_documents)
    
    with open('document_names.txt', 'r', encoding='utf-8') as f:
        document_names = [line.strip() for line in f.readlines()]
    
    # Initialize Sastrawi stopword remover and stemmer
    stopword_factory = StopWordRemoverFactory()
    stemmer_factory = StemmerFactory()
    stopword_remover = stopword_factory.create_stop_word_remover()
    stemmer = stemmer_factory.create_stemmer()

    # Input query
    query = input("Masukkan query: ")

    # Remove stopwords and perform stemming on the query
    query = stopword_remover.remove(query)
    query = stemmer.stem(query)
    
    
    start_time = time.time()
    # Search using BM25
    search_results = bm25.search(query)

    # Display search results
    print("\nHasil Pencarian:")
    for rank, (doc_index, score) in enumerate(search_results, start=1):
        document_name = document_names[doc_index]
        print(f"Rank: {rank}")
        print(f"Nama Dokumen: {document_name}")
        print(f"Skor: {score:.7f}\n")
        
end_time = time.time()

print(f"Waktu eksekusi: {end_time - start_time:.9f} detik")
