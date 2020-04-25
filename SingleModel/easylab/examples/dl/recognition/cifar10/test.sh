#!/bin/sh



#source ~/.bashrc

python cifar10.py --gpu -1 --batch-size 256 --seed 2018 --lr 0.1 --epoch 200 --net resnet --comment cifar10