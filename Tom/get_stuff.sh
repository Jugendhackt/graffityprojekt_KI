#!/usr/bin/bash

./get_urls_startpage.py urls_good.txt graffity

./get_urls_startpage.py urls_bad.txt landscape 
./get_urls_startpage.py urls_bad.txt people
./get_urls_startpage.py urls_bad.txt machines
./get_urls_startpage.py urls_bad.txt food
./get_urls_startpage.py urls_bad.txt animals

echo "Got $(cat urls_good.txt | wc -l) good samples"
echo "Got $(cat urls_bad.txt | wc -l) bad samples"

./download.py urls_good.txt data/g/
./download.py urls_bad.txt data/b/
