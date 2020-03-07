import requests
from bs4 import BeautifulSoup
import codecs

# for i in range(1, 52):
#     print("Downloading https://www.pexels.com/search/street%20art/)
#     html = requests.get("https://www.pexels.com/search/street%20art/)

f = codecs.open("pexels.html", encoding="utf-8")
html = f.read()
f.close()

soup = BeautifulSoup(html)

print(soup.prettify())