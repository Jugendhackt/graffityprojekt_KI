#!/usr/bin/bash

./get_urls_startpage.py urls_graffity.txt graffity &
./get_urls_startpage.py urls_landscape.txt landscape  &
./get_urls_startpage.py urls_people.txt people &
./get_urls_startpage.py urls_machines.txt machines &
./get_urls_startpage.py urls_food.txt food &
./get_urls_startpage.py urls_animals.txt animals &
wait

./download.py urls_graffity.txt data/graffity/ &
./download.py urls_landscape.txt data/landscape/ &
./download.py urls_people.txt data/people/ &
./download.py urls_machines.txt data/machines/ &
./download.py urls_food.txt data/food/ &
./download.py urls_animals.txt data/animals/ &
wait
