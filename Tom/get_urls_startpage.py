#!/usr/bin/python3

import requests
from bs4 import BeautifulSoup
import codecs

urls = []

for i in range(1, 29):
    html = requests.get("https://www.startpage.com/sp/search?t=default&lui=english&language=english&query=graffity&cat=pics&page=" + str(i) + "&sc=GC55zKb1Xg1J20&image-size-select=isz%3Alt%2Cislt%3Avga")

    soup = BeautifulSoup(html.content, features="html5lib")

    imgs = soup.findAll("img")

    for img in imgs:
        try:
            if "url" in img["src"]:
                urls.append(img["src"])
        except:
            pass

f = open("urls.txt", "a")
for u in urls:
    f.write(str(u) + "\r\n")

f.close()
