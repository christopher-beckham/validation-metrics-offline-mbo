#!/bin/bash

# good seeds: 11

cd ..

if [ -z $1 ]; then
  echo "Error: must specify experiment folder (relative to save dir)"
  echo "Usage: bash eval-plot.sh <experiment name> <optional extra args>"
  exit 1
fi

NAME=${SAVEDIR}/$1
OUTPUT=/tmp/output

echo "Processing " $folder

python eval_plot.py \
  --experiment="${NAME}/$folder" \
  --savedir=$SAVEDIR \
  "${@:2}"
