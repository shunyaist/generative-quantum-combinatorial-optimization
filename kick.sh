#!/bin/bash
## Usage: ./kick_solo.sh 0 1 Transformer

JOB_ID=$1
num_hosts=$2
encoder_type='Transformer'

source ./options.sh $JOB_ID $num_hosts $encoder_type

## Run
python3.11 gqco_main.py \
    ${job_options} \
    ${quantum_options} \
    ${model_options} \
    ${trainning_option} \
    ${gate_option} \
    ${wandb_option}
