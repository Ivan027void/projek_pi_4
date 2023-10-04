import requests
from bs4 import BeautifulSoup

def get_links(url):
  response = requests.get(url)
  soup = BeautifulSoup(response.content, 'html.parser')
  links = soup.find_all('a', href=True)

  return links

def main():
  # Loop through the dates
  
  month_list =['01','02','03','04','05','06','07','08','09','10','11','12']
  for date in range(2020, 2021):
    for month in month_list:
      for day in range(1, 31):
        # Loop through the pages
        for page in range(1, 31):
          # Create the URL
          url = 'https://www.kompasiana.com/indeks?page={}&category=&date={}-{}-{}'.format(page, date,month,day)

          # Get the links
          links = get_links(url)

          # Write the links to a file
          with open('links.txt', 'a') as f:
            for link in links:
              f.write(link['href'] + '\n')

if __name__ == '__main__':
  main()
