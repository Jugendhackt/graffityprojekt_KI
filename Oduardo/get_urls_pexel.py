#!/usr/bin/python3

import requests
from bs4 import BeautifulSoup
import codecs

urls = []

f = codecs.open("pexels.html", encoding="utf-8")
html = f.read()
f.close()

soup = BeautifulSoup(html, features="html5lib")

imgs = soup.findAll("img")

for img in imgs:
    if ".jpeg" in img["src"]:
        urls.append(img["src"].split("?")[0])

f = open("urls.txt", "a")
for u in urls:
    f.write(str(u) + "\r\n")

f.close()
