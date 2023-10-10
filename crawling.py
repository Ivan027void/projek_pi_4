import requests
from bs4 import BeautifulSoup

def get_urls(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    links = []
    for article in soup.find_all('h2', class_='title'):
        link = article.find('a')['href']
        links.append(link)

    return links

def save_urls(links):
    with open('links.txt', 'a') as f:
        for link in links:
            f.write(link + '\n')

if __name__ == '__main__':
    # Buat URL pencarian
    year = 2021
    for month in range(1, 12):
        for day in range(1, 31):
            date = '{}-{}-{}'.format(year, month, day)
            url = 'https://www.kompasiana.com/indeks?page=1&category=&date={}-{}-{}'.format(date, month, day)

            # Ambil semua URL
            links = get_urls(url)

            # Simpan URL ke file
            save_urls(links)
            print('links berhasil')
