#!/bin/bash

# for seed in 373 0 42
for seed in 42
do
  # for t in 1.0 2.0
  # for t in 2.0 1.0 1.5
  for t in 2.0
  do
    # for c in 1 5 10 20 100
    for c in 100
    do
      for s in 3 4 5 6 7 8 9 10
      # for s in 9 10
      do
          python3.11 test_gqco.py --temperature $t --num-clone $c --size $s --seed $seed
      done    
    done
  done  
done