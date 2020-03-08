#!/usr/bin/bash

if [ "$#" -lt 2 ]
then
	exit 1
fi

while read line
do
	for i in {1..4}
	do
		curl "$line" --output "$(mktemp -p $2 XXXXXXXXXXX.png)" &
	done
	wait
done < "$1"
