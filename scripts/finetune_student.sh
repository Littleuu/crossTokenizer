#! /bin/bash
export TOKENIZERS_PARALLELISM=false;
export TASK="sft";
export BASE_PATH=$PWD;
export MODEL_TYPE="gpt2";
export CKPT_PATH=openai_gpt2_120M;
export DATA_NAME="CodeM";  # "CodeM" for code generation task; "dolly" for instruction following task; "metamath" for math reasoning task
export DATA_DIR=$BASE_PATH/data/${DATA_NAME};
export BATCH_SIZE=4;
export LR=2e-5;  
export GRAD_ACC=2;
export EPOCH=3;
export MAX_LENGTH=512;
export SEED=6;
export SAVE_PATH=${BASE_PATH}/finetune/${DATA_NAME}/${MODEL_TYPE}/E${EPOCH}_B${BATCH_SIZE}x${GRAD_ACC}_lr${LR};
export CUDA_VISIBLE_DEVICES=1;

torchrun --nproc_per_node 1 \
    --nnodes 1 --node_rank 0 --master_addr localhost --master_port 4345 \
    ${BASE_PATH}/distillation.py \
    --base-path ${BASE_PATH} --model-path ${CKPT_PATH} --model-type ${MODEL_TYPE} \
    --n-gpu 1 --data-dir ${DATA_DIR} --num-workers 0 --dev-num 300 \
    --task ${TASK} --lr ${LR} --batch-size ${BATCH_SIZE} \
    --eval-batch-size 4 --clip-grad 1.0 \
    --gradient-accumulation-steps ${GRAD_ACC} --warmup-iters 0 \
    --lr-decay-style cosine --weight-decay 1e-2 \
    --num-epochs ${EPOCH} --max-length ${MAX_LENGTH} \
    --max-prompt-length 256 --do-train --do-valid \
    --eval-gen --save-interval 1 --eval-interval 1 --log-interval 50 \
    --save-dir ${SAVE_PATH} --keep-best-n-checkpoints 1 \
    --criterion cross_entropy --seed ${SEED} \
    --deepspeed --deepspeed_config ${BASE_PATH}/configs/deepspeed/ds_config_zero2_bf16.json \
    --do-sample --top-k 0 --top-p 1.0 --temperature 1.0;

