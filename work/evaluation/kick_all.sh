#!/bin/bash

## GQCO
for seed in 42
  do
  # for t in 1.0 2.0
  for t in 2.0
  do
    # for c in 1 5 10 20 100
    for c in 1 5 10 20 100
    do
      # for s in 3 4 5 6 7 8 9 10
      for s in 9 10
      do
          python3.11 test_gqco.py --temperature $t --num-clone $c --size $s --seed $seed
      done    
    done
  done  
done


## SA
# for r in 10
# do
#   for sw in 100 1000 10000 100000 1000000 10000000
#   do
#     for s in 3 4 5 6 7 8 9 10
#     do
#         python3.11 test_sa.py --num-sweeps $sw --num-reads $r --size $s
#     done    
#   done
# done


## QAOA
for l in 1 2 3 4 5
do
  for s in 3 4 5 6 7 8 9 10
  do
      python3.11 test_qaoa.py --layer-count $l --size $s
  done    
done