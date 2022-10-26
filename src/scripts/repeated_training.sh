#!/bin/bash
# {0..9}
for i in 0 
do
   python src/models/train_model.py --gpu-id ${2} --split-id $i --config ${1} --memory-limit ${3}
done