# cd src/open-r1-multimodal
# src/virft/src/open_r1/grpo_gui_grounding_lisa.py
export DEBUG_MODE="true"
export LOG_PATH="./debug_log_2b.txt"


# 新增：禁用 NCCL 共享内存，改用套接字通信 下面4个我加的禁用nccl
# export NCCL_SHM_DISABLE=1
# export NCCL_IB_DISABLE=1
# export NCCL_SOCKET_IFNAME=lo
# export NCCL_DEBUG=WARN  # 仅输出警告，减少日志干扰

torchrun --nproc_per_node="4" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12346" \
    src/virft/src/open_r1/grpo_lisa.py \
    --output_dir "out/lisa_train_GIoU_7b" \
    --model_name_or_path /inspire/hdd/global_user/zhouwei-240108540164/autodrive/Visual-RFT-main/Qwen/Qwen2-VL-7B-Instruct \
    --dtype float16 \
    --dataset_name NOT_USED \
    --deepspeed /inspire/hdd/global_user/zhouwei-240108540164/autodrive/Visual-RFT-main/src/virft/local_scripts/zero3.json \
    --max_prompt_length 1024 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --logging_steps 1 \
    --bf16 "true" \
    --gradient_checkpointing true \
    --attn_implementation flash_attention_2 \
    --max_pixels 401408 \
    --num_train_epochs 6 \
    --run_name Qwen2-VL-2B-GRPO-groud_lisa_train \
    --save_steps 50 \
    --eval_strategy "no" \
    --save_only_model true \
    --num_generations 4  # 8原来是需要能被整除上面的步数和gpu的数量2x2=4 4不能除8 number of outputs G in grpo, reduce it would lead to faster training and smaller memory cost but higher variance  
    
