#! /bin/bash
export TOKENIZERS_PARALLELISM=false;
export BASE_PATH=$PWD;
export CKPT_TYPE="opt";
export CKPT_PATH=facebook/opt-125m;
export TEACHER_MODEL_TYPE="pythia";
export TEACHER_MODEL_PATH=databricks/dolly-v2-3b;
# export teacher_peft_path=if needed, add --teacher-peft-path $teacher_peft_path to the command;
export TASK="multiOT";
export CRITERION="multi_level_OT";
export EPOCH=7;
export BATCH_SIZE=4;
export LR=1e-4;
export GRAD_ACC=2;
export KD_TEMP=2.0;
export MAX_LENGTH=512;
export dtype="bf16";
export SEED=10;
export KD_RATE=1.5;
export train_num=-1;
export DATA_NAME=dolly;  # "CodeM" for code generation task; "dolly" for instruction following task; "metamath" for math reasoning task
export save_model_path=${BASE_PATH}/outputs/${DATA_NAME}/${TEACHER_MODEL_TYPE}_${CKPT_TYPE}/${TASK}/${dtype}_r${KD_RATE}_e${EPOCH}_b${BATCH_SIZE}x${GRAD_ACC}_lr${LR};
export CUDA_VISIBLE_DEVICES=1;

# distillation
export DATA_DIR=${BASE_PATH}/data/${DATA_NAME};
torchrun --nproc_per_node 1 \
    --nnodes 1 --node_rank 0 --master_addr localhost --master_port 1347 \
    ${BASE_PATH}/distillation.py \
    --base-path ${BASE_PATH} --model-type ${CKPT_TYPE} \
    --model-path ${CKPT_PATH} --n-gpu 1 --train-num ${train_num} --dev-num 300 \
    --teacher-model-type ${TEACHER_MODEL_TYPE} \
    --teacher-model-path ${TEACHER_MODEL_PATH} \
    --teacher-model-fp16 --gradient-checkpointing --num-workers 0 \
    --data-dir ${DATA_DIR} --task ${TASK} --model-dtype $dtype \
    --lr ${LR} --batch-size ${BATCH_SIZE} --eval-batch-size 8 \
    --gradient-accumulation-steps ${GRAD_ACC} --warmup-iters 0 --lr-decay-style cosine \
    --weight-decay 1e-2 --clip-grad 1.0 --num-epochs ${EPOCH} --kd-rate ${KD_RATE} \
    --kd-temperature ${KD_TEMP} --max-length ${MAX_LENGTH} --max-prompt-length 256 \
    --do-train --do-valid --eval-gen --save-interval 1 --eval-interval 1 --log-interval 50 \
    --save-dir ${save_model_path} --keep-best-n-checkpoints 1 --seed ${SEED} --criterion ${CRITERION} \
    --deepspeed_config ${BASE_PATH}/configs/deepspeed/ds_config_${dtype}.json \
    --do-sample --top-k 0  --top-p 1.0 --temperature 1.0; 

# eval datasets
# instruction following: ["dolly" "self-inst" "vicuna" "snist" "unist"]
# code generation: ["humaneval"]
# math reasoning: ["metamath" "gsm8k"  "math" "orca"]
export DATA_NAMES=("dolly");  
export CKPT_PATH=${save_model_path}/epoch*;
export DATA_NUM=-1;  
export EVAL_BATCH_SIZE=4;
export eval_TASK="eval_main";

for DATA_NAME in "${DATA_NAMES[@]}"
do
    export DATA_DIR=${BASE_PATH}/data/${DATA_NAME};
    export output_PATH=${save_model_path}/eval_${DATA_NAME};

    torchrun --nproc_per_node 1 \
        --nnodes 1 --node_rank 0 --master_addr localhost --master_port 1314 \
        ${BASE_PATH}/evaluate_code_generation.py \
        --base-path ${BASE_PATH} --model-path ${CKPT_PATH} \
        --n-gpu 1 --model-type ${CKPT_TYPE} --task ${eval_TASK} \
        --data-dir ${DATA_DIR} --data-names ${DATA_NAME} --num-workers 0 \
        --dev-num ${DATA_NUM} --data-process-workers -1 --json-data \
        --eval-batch-size ${EVAL_BATCH_SIZE} --max-length 1024 --max-prompt-length 256 \
        --do-eval --save-dir ${output_PATH} --seed ${SEED} --deepspeed \
        --deepspeed_config ${BASE_PATH}/configs/deepspeed/ds_config_bf16.json \
        --do-sample --top-k 0 --top-p 1.0 --temperature 1.0;  

done