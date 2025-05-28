#!/bin/bash

timestamp=$(date +%F_%H-%M-%S)
epochs=30
batch=0.6

echo $timestamp $epochs $batch

python finetune_with_mlflow.py --epochs $epochs --batch $batch > finetune_${epochs}_${batch}_${timestamp}.log 2>&1 &