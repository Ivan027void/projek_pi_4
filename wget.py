import requests
from bs4 import BeautifulSoup

def get_all_article_urls(url):
    urls = []
    page = 1
    
    while True:
        # Mengirim permintaan GET ke halaman indeks dengan nomor halaman tertentu
        response = requests.get(f"{url}?page={page}")
        soup = BeautifulSoup(response.content, 'html.parser')

        # Mencari semua tautan ('a') dalam halaman
        for a_tag in soup.find_all('a', href=True):
            href = a_tag['href']
            if href.startswith('https://www.kompasiana.com/'):
                urls.append(href)

        # Jika tidak ada tautan lagi di halaman, keluar dari loop
        if not soup.find('a', class_='next'):
            break

        page += 1
    
    return urls

# URL halaman indeks Kompasiana
index_url = 'https://www.kompasiana.com/indeks'

# Panggil fungsi untuk mengambil URL
article_urls = get_all_article_urls(index_url)

# Cetak semua URL yang telah diambil
for url in article_urls:
    print(url)
