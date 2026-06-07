export DATA_PATH=/inspire/hdd/global_user/zhouwei-240108540164/autodrive/Visual-RFT-main/share_data/ViRFT_COCO_base65
export CKPT_PATH=/inspire/hdd/global_user/zhouwei-240108540164/autodrive/Visual-RFT-main/out/Qwen2-VL-2B-Instruct_GRPO_coco_base65cate_6k/checkpoint-850test
export SAVE_PATH=/inspire/hdd/global_user/zhouwei-240108540164/autodrive/Visual-RFT-main/out/Qwen2-VL-2B-Instruct_GRPO_coco_base65cate_6k_test_lr

export DEBUG_MODE="true" # Enable Debug if you want to see the rollout of model during RL
export LOG_PATH="./debug_log_2b_GRPO_coco_base65cate_6k.txt"


torchrun --nproc_per_node="4" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12345" \
    src/virft/src/open_r1/grpo.py \
    --output_dir ${SAVE_PATH}  \
    --model_name_or_path ${CKPT_PATH} \
    --dataset_name ${DATA_PATH} \
    --deepspeed /inspire/hdd/global_user/zhouwei-240108540164/autodrive/Visual-RFT-main/src/virft/local_scripts/zero3.json \
    --max_prompt_length 1024 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --logging_steps 1 \
    --bf16 "true" \
    --report_to wandb \
    --gradient_checkpointing false \
    --attn_implementation flash_attention_2 \
    --max_pixels 401408 \
    --num_train_epochs 2 \
    --run_name Qwen2-VL-2B_GRPO_coco_base65cate_6k \
    --save_steps 50 \
    --save_only_model true \
    --num_generations 4 \
    --learning_rate 0  