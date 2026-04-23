#! /bin/bash
export TOKENIZERS_PARALLELISM=false;
export BASE_PATH=$PWD;
export CKPT_TYPE="opt";
export CKPT_PATH=/data/user/whx/CTKD/SEDI/finetune/dolly/opt/E10_B4x2_lr2e-5/epoch9_step12861_loss2.6302_rougel23.8910
export TEACHER_MODEL_TYPE="llama";
export TEACHER_MODEL_PATH=/data/newdisk/whx/model_hub/llama2/llama2-7b-hf
export TEACHER_PEFT_PATH=/data/newdisk/whx/llama2/llama2-7b-hf/sft/criterion=cross_entropy__lora-rank=256-alpha=8-dropout=0.1-bf16__epoch=10__bsz=8x1x2=16__lr=0.001/epoch10_step7150_loss3.0127_rougel33.4382
export STUDENT_MODEL_FOR_SFT=/data/user/whx/CTKD/SEDI/finetune/dolly/opt/E10_B4x2_lr2e-5/epoch9_step12861_loss2.6302_rougel23.8910
# export teacher_peft_path=if needed, add --teacher-peft-path $teacher_peft_path to the command;
export TASK="SFT_SEDI";   
export CRITERION="SFT_SEDI_distillation";
export EPOCH=7;
export BATCH_SIZE=4;
export LR=1e-4;
export GRAD_ACC=2;
export KD_RATE=0.5;
export KD_TEMP=2.0;
export MAX_LENGTH=512;
export dtype="bf16";  
export SEED=10;
export K=100;  
export train_num=-1;
export CUDA_VISIBLE_DEVICES=1;

# distillation
export DATA_NAME=dolly;  # "CodeM" for code generation task; "dolly" for instruction following task; "metamath" for math reasoning task
export DATA_DIR=${BASE_PATH}/data/${DATA_NAME};
export save_model_path=${BASE_PATH}/outputs/$DATA_NAME/${TEACHER_MODEL_TYPE}_${CKPT_TYPE}/${TASK}/${dtype}_r${KD_RATE}_e${EPOCH}_b${BATCH_SIZE}x${GRAD_ACC}_lr${LR}_K${K};
torchrun --nproc_per_node 1 \
    --nnodes 1 --node_rank 0 --master_addr localhost --master_port 2946\
    ${BASE_PATH}/distillation.py \
    --base-path ${BASE_PATH} --model-type ${CKPT_TYPE} \
    --model-path ${CKPT_PATH} --n-gpu 1 --train-num ${train_num} --dev-num 500 \
    --student-model-for-sft ${STUDENT_MODEL_FOR_SFT} \
    --teacher-model-type ${TEACHER_MODEL_TYPE} \
    --teacher-model-path ${TEACHER_MODEL_PATH} \
    --teacher-peft-path ${TEACHER_PEFT_PATH} \
    --teacher-model-fp16 --gradient-checkpointing --num-workers 0 \
    --data-dir ${DATA_DIR} --task ${TASK} --model-dtype $dtype \
    --lr ${LR} --batch-size ${BATCH_SIZE} --eval-batch-size 8 \
    --gradient-accumulation-steps ${GRAD_ACC} --warmup-iters 0 --lr-decay-style cosine \
    --weight-decay 1e-2 --clip-grad 1.0 --num-epochs ${EPOCH} --kd-rate ${KD_RATE} \
    --kd-temperature ${KD_TEMP} --max-length ${MAX_LENGTH} --max-prompt-length 256 \
    --do-train --do-valid --eval-gen --save-interval 1 --eval-interval 1 --log-interval 50 \
    --save-dir ${save_model_path} --keep-best-n-checkpoints 1 --seed ${SEED} --criterion ${CRITERION} \
    --deepspeed_config ${BASE_PATH}/configs/deepspeed/ds_config_${dtype}.json \
    --do-sample --top-k 0  --top-p 1.0 --temperature 1.0 --K $K;   


# eval datasets
# instruction following: ["dolly" "self-inst" "vicuna" "snist" "unist"]
# code generation: ["humaneval"]
# math reasoning: ["metamath" "gsm8k"  "math" "orca"]
# export DATA_NAMES=("unist");  
# export CKPT_PATH=/data/user/whx/CTKD/SEDI/outputs/dolly/llama2_opt/SFT_SEDI/bf16_r0.5_e7_b4x2_lr1e-4_K100/epoch7_step10003_loss4.0551_rougel26.4315
# export DATA_NUM=-1; 
# export EVAL_BATCH_SIZE=4;
# export eval_TASK="eval_main";
# for DATA_NAME in "${DATA_NAMES[@]}"
# do
#     export DATA_DIR=${BASE_PATH}/data/${DATA_NAME};
#     export output_PATH=${save_model_path}/eval_${DATA_NAME};

#     torchrun --nproc_per_node 1 \
#         --nnodes 1 --node_rank 0 --master_addr localhost --master_port 8121 \
#         ${BASE_PATH}/evaluate_dolly.py \
#         --base-path ${BASE_PATH} --model-path ${CKPT_PATH} \
#         --n-gpu 1 --model-type ${CKPT_TYPE} --task ${eval_TASK} \
#         --data-dir ${DATA_DIR} --data-names ${DATA_NAME} --num-workers 0 \
#         --dev-num ${DATA_NUM} --data-process-workers -1 --json-data \
#         --eval-batch-size ${EVAL_BATCH_SIZE} --max-length 512 --max-prompt-length 256 \
#         --do-eval --save-dir ${output_PATH} --seed ${SEED} --deepspeed \
#         --deepspeed_config ${BASE_PATH}/configs/deepspeed/ds_config_infer_bf16.json \
#         --do-sample --top-k 0 --top-p 1.0 --temperature 1.0;
# done