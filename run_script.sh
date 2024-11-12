#!/bin/bash

counter=1

while true; do
	filename="training_set_1_2.${counter}.json"
	echo "\n----------\nStarting training ${counter}\n----------\n"
	python ammonoid_finetuning_llama_3_1_8b.py "$filename"

	if [ $? -ne 0]; then
		echo "Python script encountered an error. Stopping."
		break
	fi

	((counter++))
	sleep 1
done

