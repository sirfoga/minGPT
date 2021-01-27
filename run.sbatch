#!/bin/bash

#SBATCH --partition=ml
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=4096
#SBATCH --time=6:00:00
#SBATCH --account=p_ml_cv
#SBATCH --output=%j.out
#SBATCH --error=%j.err

echo "++++++++++ resources granted"

module --force purge
module load modenv/ml
module load Python/3.7
module load PyTorch
module load scikit-learn
module load matplotlib
module load h5py/2.10

source .venv/bin/activate

echo "++++++++++ requirements ok"
echo "using $(which python)"  # just to check

rm -rf results/
python play_image.py

exit 0
