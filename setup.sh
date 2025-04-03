#!/bin/bash

device=$1

python3.11 -m venv .env

source .env/bin/activate

if [ "$device" == "gpu" ]; then
    module load cuda/11.8/11.8.0 || true
    module load cudnn/8.9/8.9.7 || true
    module load nccl/2.16/2.16.2-1 || true
    module load hpcx/2.12 || true
    requirements_file="requirements.txt"
else
    requirements_file="requirements_cpu.txt"
fi

pip install --upgrade pip setuptools
pip install -r $requirements_file

pip install ipykernel
python -m ipykernel install --user --name .env --display-name "Python (.env)"

echo "Setup completed successfully!"
