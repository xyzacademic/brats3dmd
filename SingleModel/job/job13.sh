#!/bin/sh

#$ -q datasci
#$ -q datasci3
#$-cwd
#$-N v15

cd ..



python single_train.py --gpu -1 --data 2 --norm all --batch-size 4 --basefilter 8 --seed 4231 --lr 0.02 --epoch 240 --fp16 --loss brats19v2 --schedule s1 --net v15 --flip --comment single_train_v15_240_f3_shift
python single_validation.py --gpu -1 --data 2 --norm all --batch-size 8 --basefilter 8 --seed 4096 --lr 0.01 --epoch 200 --fp16 --loss brats19v2 --schedule s1 --net v15 --resume --comment single_train_v15_240_f3_shift
