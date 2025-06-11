#!/bin/bash

timestamp=$(date +%F_%H-%M-%S)
epochs=30
batch=16

echo $timestamp $epochs $batch

python finetune_with_mlflow.py --epochs $epochs --batch $batch > logs/finetune_${epochs}_${batch}_${timestamp}.log 2>&1 &
