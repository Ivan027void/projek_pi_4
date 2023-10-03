import requests
from bs4 import BeautifulSoup

def get_article_urls(url):
    urls = []
    
    page = 1
    while True:
        # Mengirim permintaan GET ke halaman indeks dengan nomor halaman tertentu
        response = requests.get(f"{url}?page={page}")
        soup = BeautifulSoup(response.content, 'html.parser')
        
        response = requests.get(url)
        soup = BeautifulSoup(response.content, "html.parser")

        for urls in soup.find_all('a'):
            if urls.get('href') and urls.get('href').startswith('https://www.kompasiana.com/'):
                urls.append(urls.get('href'))
        
        page += 1
    
    return urls

# URL halaman indeks Kompasiana
index_url = 'https://www.kompasiana.com/indeks'

# Panggil fungsi untuk mengambil URL
article_urls = get_article_urls(index_url)

# Cetak semua URL yang telah diambil
for url in article_urls:
    print(url)
