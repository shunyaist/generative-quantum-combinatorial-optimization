#!/bin/bash

# for l in 1 2 3 4 5
for l in 2
do
  for s in 4
  # 4 5 6 7 8 9 10
  do
      python3.11 test_qaoa.py --layer-count $l --size $s
  done    
done