srun \
    --partition=ml \
    --nodes=1 \
    --tasks=1 \
    --cpus-per-task=2 \
    --gres=gpu:1 \
    --mem-per-cpu=4096 \
    --time=02:00:00 \
    --account=p_ml_cv \
    --pty bash

KERNEL_NAME="pytorch"
KERNELS_DIR=/home/${USER}/kernels/

module --force purge
module load modenv/ml
module load Python/3.7
module load PyTorch
module load scikit-learn
module load matplotlib
module load h5py/2.10

source .venv/bin/activate
ipython
