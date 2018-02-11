#!/usr/bin/env bash

mkdir -p dataset/

wget https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz -O dataset/cifar-10-python.tar.gz

cd dataset
dtrx --one=here -n cifar-10-python.tar.gz
