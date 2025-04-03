#!/bin/bash
#$ -l h_rt=01:00:00
#$ -l rt_F=32
#$ -j y
#$ -o outputs/
#$ -cwd   


## Directory and environment
cd <your directory>
source .env/bin/activate

# Module
source /etc/profile.d/modules.sh
module load cuda/11.8/11.8.0
module load cudnn/8.9/8.9.7
module load nccl/2.16/2.16.2-1
module load hpcx/2.12


## Options
source ./options.sh ${JOB_ID} ${NHOSTS}


## Run
export OMP_NUM_THREADS=20
export NGPU_PER_NODE=4

# launch on slave nodes
node_rank=1
for slave_node in `cat $SGE_JOB_HOSTLIST | awk 'NR != 1 { print }'`; do
qrsh -inherit -V -cwd $slave_node fabric run \
    --strategy ddp \
    --accelerator cuda \
    --devices $NGPU_PER_NODE \
    --num-nodes $NHOSTS \
    --node-rank $node_rank \
    --main-address `hostname` \
    gqco_main.py \
    ${job_options} \
    ${quantum_options} \
    ${model_options} \
    ${training_option} \
    ${gate_option} \
    ${wandb_option} &
node_rank=`expr $node_rank + 1`
done

# launch on master node
node_rank=0
fabric run \
    --strategy ddp \
    --accelerator cuda \
    --devices $NGPU_PER_NODE \
    --num-nodes $NHOSTS \
    --node-rank $node_rank \
    --main-address `hostname` \
    gqco_main.py \
    ${job_options} \
    ${quantum_options} \
    ${model_options} \
    ${training_option} \
    ${gate_option} \
    ${wandb_option}


# finalize
wait
exit 0
