#!/bin/bash
dataset=$1
subject=$2  

echo "######################## First Stage "########################
python train.py -s dataset/parsed/$dataset/$subject -m output/$dataset/$subject --quiet

echo "######################## Second Stage "########################
python merge.py -s dataset/parsed/$dataset/$subject -m output/$dataset/$subject --quiet

echo "######################## Third Stage "########################
python train.py -s dataset/parsed/$dataset/$subject -m output/$dataset/$subject --quiet
