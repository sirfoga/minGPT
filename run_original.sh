MODEL_SIZE="s"  # "l"
MODEL_CHECKPOINT=1000000
DOWNLOAD_DIR="."
DATA_SET="imagenet"

# prepare stuff

python download.py --model ${MODEL_SIZE} --ckpt ${MODEL_CHECKPOINT} --download_dir ${DOWNLOAD_DIR}  # pre-trained model

python download.py --model ${MODEL_SIZE} --ckpt ${MODEL_CHECKPOINT} --dataset ${DATA_SET} --download_dir ${DOWNLOAD_DIR}  # center-cropped images intended for evaluation

python download.py --model ${MODEL_SIZE} --ckpt ${MODEL_CHECKPOINT} --dataset ${DATA_SET} --clusters  # color cluster file defining our 9-bit color palette

# sampling

SAMPLING_EMBEDDING=512  # 1536
SAMPLING_HEADS=8  # 16
SAMPLING_LAYERS=24  # 48

CUDA_VISIBLE_DEVICES=0 python src/run.py --sample --n_embd ${SAMPLING_EMBEDDING} --n_head ${SAMPLING_HEADS} --n_layer ${SAMPLING_LAYERS} -ckpt_path ${DOWNLOAD_DIR} --color_cluster_path ${DOWNLOAD_DIR} --n_gpu 1

# eval loss

python src/run.py --eval --n_embd ${SAMPLING_EMBEDDING} --n_head ${SAMPLING_HEADS} --n_layer ${SAMPLING_LAYERS} -ckpt_path ${DOWNLOAD_DIR} --color_cluster_path ${DOWNLOAD_DIR}
