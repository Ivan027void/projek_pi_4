import requests
from bs4 import BeautifulSoup

def get_links(url):
  response = requests.get(url)
  soup = BeautifulSoup(response.content, 'html.parser')
  links = soup.find_all('a')
  return links

if __name__ == '__main__':
  url = 'https://www.kompasiana.com/'
  links = get_links(url)

  with open('links.txt', 'w') as f:
    for link in links:
      f.write(link['href'] + '\n')
