#!/bin/bash

# good seeds: 11

cd ..

NAME=${SAVEDIR}/$1

echo "Processing " $folder

python eval_plot.py \
--experiment="${NAME}/$folder" \
"${@:2}"
