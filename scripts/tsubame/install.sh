#!/bin/sh
#$ -cwd
#$ -l node_q=1
#$ -l h_rt=1:00:00
#$ -p -5

# priority: -5: normal, -4: high, -3: highest

# Load modules
module use /gs/fs/tga-NII-LLM/modules/modulefiles

module load ylab/cuda/12.4
module load ylab/cudnn/9.1.0
module load ylab/nccl/cuda-12.4/2.21.5
module load ylab/hpcx/2.17.1
module load ninja/1.11.1

# Set environment variables
source .env/bin/activate

pip install --upgrade pip
pip install --upgrade wheel cmake ninja packaging

# install nvidia pytorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# huggingface
pip install transformers datasets huggingface_hub

# other packages
pip install numpy
MAX_JOBS=16 pip install flash-attn==2.5.8 --no-build-isolation
