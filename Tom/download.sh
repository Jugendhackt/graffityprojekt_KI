#!/usr/bin/bash

while read line
do
	echo "Downloading $line ..."
	# curl "$line" --output "$2/$(mktemp XXXXXXXXXX.jpeg)"
	wget "$line"
done < "$1"
