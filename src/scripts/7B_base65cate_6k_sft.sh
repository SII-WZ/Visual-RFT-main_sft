export DATA_PATH=/inspire/hdd/global_user/zhouwei-240108540164/autodrive/Visual-RFT-main/share_data/COCOwothink
export CKPT_PATH=/inspire/hdd/global_user/zhouwei-240108540164/autodrive/Visual-RFT-main/Qwen/Qwen2-VL-2B-Instruct
export SAVE_PATH=/inspire/hdd/global_user/zhouwei-240108540164/autodrive/Visual-RFT-main/out/Qwen2-VL-2B-Instruct_SFT_coco_base65cate_6k_sft_wothink_new30


torchrun --nproc_per_node=4 \
--nnodes=1 \
--node_rank=0 \
--master_addr=127.0.0.1 \
--master_port=12345 \
src/virft/src/open_r1/sft.py \
--output_dir ${SAVE_PATH} \
--model_name_or_path ${CKPT_PATH} \
--dataset_name ${DATA_PATH} \
--deepspeed /inspire/hdd/global_user/zhouwei-240108540164/autodrive/Visual-RFT-main/src/virft/local_scripts/zero3.json \
--max_seq_length 2048 \
--per_device_train_batch_size 1 \
--gradient_accumulation_steps 2 \
--logging_steps 1 \
--bf16 True \
--report_to wandb \
--gradient_checkpointing True \
--attn_implementation flash_attention_2 \
--max_pixels 401408 \
--num_train_epochs 2 \
--run_name Qwen2-VL-7B_SFT_coco_base65cate_6k \
--save_steps 50 \
--save_only_model True \
--learning_rate 2e-5 \
--dataset_train_split train \
--dataset_test_split test