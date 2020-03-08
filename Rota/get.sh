#!/usr/bin/bash

if [ "$#" -lt 2 ]
then
	exit 0
fi

i=0

while read line1
do
	# while read line2
	# do
	# 	curl "http://d9417002.eu.ngrok.io/text?message=$line1$line2" --output "$2/$line1$line2.png" --silent &
	# 	let "i=i+1"

	# 	if [ "$i" -ge 1 ]
	# 	then
	# 		echo "Waiting..."
	# 		wait
	# 	fi

	curl "http://d9417002.eu.ngrok.io/text?message=$line1$line2" --output "$2/$line1$line2.png" --silent &
	let "i=i+1"

	if [ "$i" -ge 32 ]
	then
		echo "Waiting..."
		wait
	fi
done < "$1"
