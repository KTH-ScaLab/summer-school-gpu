#!/bin/bash

#execute the program
EXE=reduction
K=1024*1024*4
for i in {1..8}
do
    echo ./$EXE $K
    srun -n 1 ./$EXE $K
    K=$(($K*2))
done

