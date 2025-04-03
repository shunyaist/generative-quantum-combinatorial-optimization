#!/bin/bash

for r in 10
do
  for sw in 100 1000 10000 100000 1000000 10000000
  # for sw in 10000000
  do
    for s in 3 4 5 6 7 8 9 10
    do
        python3.11 test_sa.py --num-sweeps $sw --num-reads $r --size $s
    done    
  done
done