#!/bin/bash
set -e   # stop script immediately if a command fails

cd /Users/paul/Desktop/cs544/hw3/submission

# Train
python3 train.py --train /Users/paul/Desktop/cs544/hw3/data/train --dev /Users/paul/Desktop/cs544/hw3/data/dev
python3 train.py --train /Users/paul/Desktop/cs544/hw3/data/train --dev /Users/paul/Desktop/cs544/hw3/data/dev --glove /Users/paul/Desktop/cs544/hw3/glove.6B.100d.gz

# Predict
python3 predict.py \
    --data /Users/paul/Desktop/cs544/hw3/data/dev \
    --model blstm1.pt \
    --output dev1.out
python3 predict.py \
    --data /Users/paul/Desktop/cs544/hw3/data/dev \
    --model blstm2.pt \
    --output dev2.out
python3 predict.py \
    --data /Users/paul/Desktop/cs544/hw3/data/test \
    --model blstm1.pt \
    --output test1.out
python3 predict.py \
    --data /Users/paul/Desktop/cs544/hw3/data/test \
    --model blstm2.pt \
    --output test2.out

# Evaluate
cd /Users/paul/Desktop/cs544/hw3/eval

python3 eval.py --gold ../data/dev --pred ../submission/dev1.out
python3 eval.py --gold ../data/dev --pred ../submission/dev2.out