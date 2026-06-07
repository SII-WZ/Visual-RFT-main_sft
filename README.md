# Visual-RFT-main_sft
环境配置 bash setup.sh

文件准备 Visual-RFT-main_sft/Qwen/ 下面放 Qwen2-VL-2B-Instruct和Qwen2-VL-7B-Instruct

参考下载：hf download --repo-type model Qwen/Qwen2-VL-7B-Instruct --local-dir  ./Qwen/Qwen2-VL-7B
 
数据准备 Visual-RFT-main_sft/share_data 下面放着数据，ViRFT_COCO_base65下面放着原文中的coco数据。其中train_clean.parquet到train_clean6.parquet 是我转化的没有think的过程coco文件。  train-00000-of-00007.parquet到train-00006-of-00007.parquet是原文给的数据。share_data/coco_val

Visual-RFT-main_sft/out是输出模型的位置

推理问题出在坐标和模型不匹配

训练
导入环境路径
export PYTHONPATH=$PYTHONPATH:/inspire/hdd/global_user/zhouwei-240108540164/autodrive/Visual-RFT-main/src/virft/src
脚本均位于src/scripts中，启动脚本用下面的命令
coco数据的训练代码
bash src/scripts/7B_base65cate_6k_sft.sh  sft微调代码启动（这个是我写的）
bash src/scripts/7B_base65cate_6k.sh   rl强化学习代码启动

lisa的与原文一致没有修改（数据直接去它给的地址下载就行）
bash  src/scripts/2B_lisa_grounding.sh启动就行

脚本里面bf16 true 记得点上 不然会出问题
训练的代码在Visual-RFT-main_sft/src/virft/src/open_r1 文件夹下面



推理
模型加载的时候把torch_dtype=torch.bfloat16设进去不然会超 
lisa的推理 #和原工作一样
cd lisa_evaluation
python iou_test.py --task test_error --split 0 --split_num 1 我自己写的计算IoU 准确率并保存结果。用python merge_eval.py 弄出结果

bash Qwen2_VL_lisa_infere.sh 原文的推理
参考Visual-RFT-main_sft/lisa_evaluation/README.md 里面的处理

coco的推理
cd ./coco_evaluation
python Qwen2_VL_coco_infere.py
注意两个参数 
selected_cate  选择种类一共有80个65个是训练集里面出现的15个ood的
question  这是问题prompt 需不需要think在这里操作。

coco训练的数据的推理测试
Visual-RFT-main_sft/coco_evaluation/evl_train65_sft.py（推理）
Visual-RFT-main_sft/coco_evaluation/evl_train65_sft_map.py（计算结果）
