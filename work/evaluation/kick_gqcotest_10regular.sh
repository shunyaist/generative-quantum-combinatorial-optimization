#!/bin/bash
#PBS -q rt_HF
#PBS -l select=1
#PBS -l walltime=12:00:00
#PBS -o outputs/
#PBS -k oe
#PBS -m n
#PBS -P 

## Directory and environment
cd ${PBS_O_WORKDIR}
source .env/bin/activate

out_dir="./outputs/${PBS_JOBID}"
mkdir -p ${out_dir}

# Module
source /etc/profile.d/modules.sh
module load cuda/11.8/11.8.0
module load cudnn/9.5/9.5.1
module load nccl/2.23/2.23.4-1
module load hpcx/2.20


GPUS=(0 1 2 3 4 5 6 7)
for i in ${!GPUS[@]}; do
    GPU=${GPUS[$i]}
    CUDA_VISIBLE_DEVICES=$GPU python ./work/evaluation/700_PerformanceFor10Regular.py --kick-id ${GPUS[$i]} >& "${out_dir}/log_${GPU}.log" &
    sleep 5
done
wait
