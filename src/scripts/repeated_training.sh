#!/bin/bash
for i in {0..9}
do
   docker exec -w /workspaces/3d_bispectrum_cnn -u vscode cool_poitras python src/models/train_model.py --gpu-id ${2} --split-id $i --config ${1} --memory-limit ${3}
done