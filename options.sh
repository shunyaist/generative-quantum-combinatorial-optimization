#!/bin/bash

JOB_ID=$1
num_hosts=$2
encoder_type='Transformer'



## Settings ------------------------------------------------------

## Task setting
### For base model training
task_name="${encoder_type}-ABCI"
task_type='base'
num_qubit=20
max_size=20
start_size=3
tune_size=-1


## For expert-tuning
# task_type='tune'
# start_size=9
# tune_size=9

## ---------------------------------------------------------------


### Job setting
seed=373
# is_wandb='True'
is_wandb='True'
project_name='GQCO'

    
### Quantum setting
quantum_tool='qiskit'
# quantum_tool='cudaq'
num_shot=-1


### Model setting
min_generation=3
max_generation=0
hidden_dim=256
encoder_hidden_dim_inner=256
encoder_depth=12
encoder_num_heads=8
decoder_num_heads=8
decoder_depth=12
decoder_ffn_dim=1024
max_size=20


## Training setting
init_checkpoint='None'

learning_rate=1e-4
max_epoch=1000000
loss_mode='one-to-all'
loss_type='dpo'
ipo_beta=0.1
dpo_beta=0.1
num_policy_search=-1


## Log setting
log_freq=10
log_freq_acc=10
checkpoint_freq=10

##----------------------------------------------------------------




## Other settings
out_dir="./outputs/${JOB_ID}"
checkpoint_dir="${out_dir}/checkpoints"

if [ "$is_wandb" = 'True' ]; then
    wandb_option="--is-wandb"
else
    wandb_option="--no-wandb"
fi


## Make directories
mkdir -p ${out_dir}
mkdir -p ${checkpoint_dir}


## Gather options
job_options=" \
    --job-id ${JOB_ID} \
    --num-hosts ${num_hosts} \
    --out-dir ${out_dir} \
    --checkpoint-dir ${checkpoint_dir} \
    --seed ${seed} \
    --project-name ${project_name} \
    --task-name ${task_name} \
    --task-type ${task_type} \
    --start-size ${start_size} \
    --tune-size ${tune_size}
"
quantum_options=" \
    --quantum-tool ${quantum_tool} \
    --num-qubit ${num_qubit}
"
model_options=" \
    --min-generation ${min_generation} \
    --max-generation ${max_generation} \
    --max-size ${max_size} \
    --hidden-dim ${hidden_dim} \
    --encoder-type ${encoder_type} \
    --encoder-hidden-dim-inner ${encoder_hidden_dim_inner} \
    --encoder-depth ${encoder_depth} \
    --encoder-num-heads ${encoder_num_heads} \
    --decoder-num-heads ${decoder_num_heads} \
    --decoder-depth ${decoder_depth} \
    --decoder-ffn-dim ${decoder_ffn_dim}
"
training_option=" \
    --learning-rate ${learning_rate} \
    --init-checkpoint ${init_checkpoint} \
    --init-checkpoint-tag ${init_checkpoint_tag} \
    --max-epoch ${max_epoch} \
    --ipo-beta ${ipo_beta} \
    --dpo-beta ${dpo_beta} \
    --num-policy-search ${num_policy_search} \
    --num-shot ${num_shot} \
    --log-freq ${log_freq} \
    --log-freq-acc ${log_freq_acc} \
    --log-freq-pretrain ${log_freq_pretrain} \
    --loss-mode ${loss_mode} \
    --loss-type ${loss_type} \
    --checkpoint-freq ${checkpoint_freq}
"
gate_option=" \
    --rot 1 \
    --rzz 1 \
    --had 1 \
    --cnot 1
"
