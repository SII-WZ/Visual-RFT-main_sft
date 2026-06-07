# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#u
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from transformers import TrainerCallback, TrainerControl, TrainerState
# 补充这行导入（关键：解决List、Dict未定义）
from typing import List, Dict, Optional
#上面都是我补充的
import torch
import os
import re
import json
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional
import traceback
from datasets import load_dataset, load_from_disk, Dataset, concatenate_datasets
from transformers import Qwen2VLForConditionalGeneration
import PIL
import numpy as np
from math_verify import parse, verify
from open_r1.trainer import Qwen2VLGRPOTrainer, Qwen2VLGRPOVLLMTrainer
from trl import GRPOConfig, GRPOTrainer, ModelConfig, ScriptArguments, TrlParser, get_peft_config
from sklearn.metrics import f1_score
from PIL import Image
#我写的验证

class EvalEvery50EpochsCallback(TrainerCallback):
    def __init__(self, eval_interval=50, log_save_path="out/lisa_train_GIoU/eval_logs.json",trainer=None):
        self.eval_interval = eval_interval  # 每50个epoch执行一次
        self.log_save_path = log_save_path  # 初始化日志保存路径
        self.trainer = trainer  # 新增：保存trainer实例为类属性
        # 确保日志目录存在
        os.makedirs(os.path.dirname(self.log_save_path), exist_ok=True)
    def on_epoch_end(self, args, state, control, **kwargs):
        """每个epoch结束时触发（仅主进程执行，修复参数不存在问题，解决生成配置冲突）"""
        # 1. 仅主进程（rank=0）执行评估，避免多进程冲突和 None 问题
        is_main_process = not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
        if (is_main_process 
            and state.epoch % self.eval_interval == 0 
            and state.epoch > 0 
            and self.trainer is not None):
            
            print(f"\n===== Epoch {state.epoch} - Running Evaluation (Main Process) =====")
            
            # 2. 关键：修改模型的生成配置（直接修改 trainer 中的模型，解决 num_return_sequences 冲突）
            # 拿到 unwrapped 模型（去除分布式包装，直接操作核心模型）
            unwrapped_model = self.trainer.model.module if hasattr(self.trainer.model, 'module') else self.trainer.model
            
            # 3. 强制设置生成配置参数，适配贪心搜索（合规，无冲突）
            unwrapped_model.generation_config.num_return_sequences = 1
            unwrapped_model.generation_config.num_beams = 1  # 明确关闭 beam search，对应贪心搜索
            unwrapped_model.generation_config.early_stopping = False
            
            # 4. 执行无参 evaluate()（Qwen2VLGRPOTrainer 支持的默认调用方式）
            eval_metrics = self.trainer.evaluate()
            
            # 5. 打印和保存日志（仅主进程，避免重复写入）
            print(f"Evaluation Metrics - Loss: {eval_metrics.get('eval_loss', 0.0):.4f}, "
                f"Reward: {eval_metrics.get('eval_reward', 0.0):.4f}, "
                f"KL: {eval_metrics.get('eval_kl', 0.0):.4f}")
            self._save_validation_results(eval_metrics)
    
    def _save_validation_results(self, val_results: dict):
        """
        追加写入验证结果，保持与训练日志一致的格式
        """
        # 补充epoch信息（可选）
        val_results["epoch"] = self.current_epoch  # 需在on_epoch_end中记录current_epoch
        with open(self.log_save_path, "a", encoding="utf-8") as f:
            json.dump(val_results, f, ensure_ascii=False)
            f.write("\n" + "="*120 + "\n")
        
        # 保存最新结果（方便快速查看）
        latest_log_path = self.log_save_path.replace(".json", "_latest.json")
        with open(latest_log_path, "w", encoding="utf-8") as f:
            json.dump(val_results, f, ensure_ascii=False, indent=2)
    
    # 新增：记录当前epoch（可选，用于日志）
    def on_epoch_start(self, args, state, control, **kwargs):
        self.current_epoch = state.epoch


#############我写的验证

@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.

    Args:
        reward_funcs (`list[str]`):
            List of reward functions. Possible values: 'accuracy', 'format'.
    """

    reward_funcs: list[str] = field(
        default_factory=lambda: ["accuracy", "format"],
        metadata={"help": "List of reward functions. Possible values: 'accuracy', 'format'"},
    )
    max_pixels: Optional[int] = field(
        default=12845056,
        metadata={"help": "Maximum number of pixels for the image"},
    )
    min_pixels: Optional[int] = field(
        default=3136,
        metadata={"help": "Minimum number of pixels for the image"},
    )

import numpy as np

def compute_giou(gt_bbox, student_bbox):
    x1_gt, y1_gt = gt_bbox[0]
    x2_gt, y2_gt = gt_bbox[1]
    
    x1_st, y1_st = student_bbox[0]
    x2_st, y2_st = student_bbox[1]

    x1_inter = max(x1_gt, x1_st)
    y1_inter = max(y1_gt, y1_st)
    x2_inter = min(x2_gt, x2_st)
    y2_inter = min(y2_gt, y2_st)

    inter_width = max(0, x2_inter - x1_inter)
    inter_height = max(0, y2_inter - y1_inter)
    inter_area = inter_width * inter_height

    gt_area = (x2_gt - x1_gt) * (y2_gt - y1_gt)
    student_area = (x2_st - x1_st) * (y2_st - y1_st)

    union_area = gt_area + student_area - inter_area

    iou = inter_area / union_area if union_area > 0 else 0

    x1_c = min(x1_gt, x1_st)
    y1_c = min(y1_gt, y1_st)
    x2_c = max(x2_gt, x2_st)
    y2_c = max(y2_gt, y2_st)

    c_area = (x2_c - x1_c) * (y2_c - y1_c)

    giou = iou - (c_area - union_area) / c_area if c_area > 0 else iou

    giou_scaled = giou + 1
    return giou_scaled



def accuracy_reward(completions, solution, **kwargs):
    """Reward function that checks if the completion is correct using either symbolic verification or exact string matching."""
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    for content, sol in zip(contents, solution):
        reward = 0.0
        # Try symbolic verification first
        try:
            answer = parse(content)
            if float(verify(answer, parse(sol))) > 0:
                reward = 1.0
        except Exception:
            pass  # Continue to next verification method if this fails

        # If symbolic verification failed, try string matching
        if reward == 0.0:
            # try:
            # Extract answer from solution if it has think/answer tags
            sol_match = re.search(r'<answer>(.*?)</answer>', sol)
            ground_truth = sol_match.group(1).strip() if sol_match else sol.strip()
            
            # Extract answer from content if it has think/answer tags
            content_match = re.search(r'<answer>(.*?)</answer>', content, re.DOTALL)
            student_answer = content_match.group(1).strip() if content_match else content.strip()
            
            try:
                pattern = r"\(\s*(\d+)\s*,\s*(\d+)\s*\)\s*,\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)"
                student_matches = re.findall(pattern, student_answer)
                student_bbox = [(int(x1) / 1000, int(y1) / 1000) for x1, y1, x2, y2 in student_matches] + [(int(x2) / 1000, int(y2) / 1000) for x1, y1, x2, y2 in student_matches]
                gt_matches = re.findall(pattern, ground_truth)
                gt_bbox = [(int(x1) / 1000, int(y1) / 1000) for x1, y1, x2, y2 in gt_matches] + [(int(x2) / 1000, int(y2) / 1000) for x1, y1, x2, y2 in gt_matches]
                reward = compute_giou(gt_bbox, student_bbox)

            except Exception as e:
                # print(traceback.format_exc())
                reward = 0.0
            # Compare the extracted answers
            if student_answer == ground_truth:
                reward = 2.0

        rewards.append(reward)
        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            # local_rank = int(os.getenv("LOCAL_RANK", 0))
            with open(log_path, "a",encoding="utf-8") as f:
                f.write(f"------------- {current_time} Accuracy reward: {reward} -------------\n")
                f.write(f"Content: {content}\n")
                f.write(f"Solution: {sol}\n")
    return rewards


def format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.fullmatch(pattern, content, re.DOTALL) for content in completion_contents]
    return [1.0 if match else 0.0 for match in matches]

reward_funcs_registry = {
    "accuracy": accuracy_reward,
    "format": format_reward,
}

SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)


def main(script_args, training_args, model_args):
    # Get reward functions
    reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs]

    dataset_ini = Dataset.from_json("/inspire/hdd/global_user/zhouwei-240108540164/autodrive/Visual-RFT-main/share_data/lisa_train_sft.json")

    # Format into conversation
    # def make_conversation(example):
    #     return {
    #         "prompt": [
    #             {"role": "system", "content": SYSTEM_PROMPT},
    #             {"role": "user", "content": example["problem"]},
    #         ],
    #     }

    QUESTION_TEMPLATE = "{Question} Output the thinking process in <think> </think> and your grouding box. Following \"<think> thinking process </think>\n<answer>(x1,y1),(x2,y2)</answer>)\" format."

    # def make_conversation(example):
    #     conversations = example["conversations"]
    #     formatted_conversation = []
    #     for message in conversations:
    #         if message["from"] == "user":
    #             formatted_conversation.append({"role": "user", "content": message["value"]})
    #         elif message["from"] == "assistant":
    #             formatted_conversation.append({"role": "assistant", "content": message["value"]})
        
    #     return {"prompt": formatted_conversation}

    def make_conversation_image(example):
        conversations = example["conversations"]
        formatted_conversation = []
        for message in conversations:
            if message["from"] == "user":
                # Check if the message contains an image
                if "<img>" in message["value"] and "</img>" in message["value"]:
                    image_path = message["value"].split("<img>")[1].split("</img>")[0].strip()
                    text_content = message["value"].replace(f"<img>{image_path}</img>", "").strip()
                    formatted_conversation.append(
                        {
                            "role": "user",
                            "content": [
                                {"type": "image"},
                                {"type": "text", "text": QUESTION_TEMPLATE.format(Question=text_content)},
                            ],
                        }
                    )
                else:
                    formatted_conversation.append({"role": "user", "content": message["value"]})
            elif message["from"] == "assistant":
                res = '<answer> ' + message["value"] + ' </answer>'
                # formatted_conversation.append({"role": "assistant", "content": message["value"]})
        # print(image_path)
        return {"image": Image.open(image_path), "prompt": formatted_conversation, 'solution': res}

    dataset = dataset_ini.map(make_conversation_image)

    
   ###我修改的
    reward_funcs_names = script_args.reward_funcs  # ["accuracy", "format"]
    reward_funcs = [reward_funcs_registry[func] for func in reward_funcs_names]
    
    dataset_val_ini = Dataset.from_json("/inspire/hdd/global_user/zhouwei-240108540164/autodrive/Visual-RFT-main/share_data/lisa_train_sft.json")
    # ... 原有数据ini处理逻辑（make_conversation_image等） ...
    dataset_val = dataset_val_ini.map(make_conversation_image)
    
    dataset_test_ini = Dataset.from_json("/inspire/hdd/global_user/zhouwei-240108540164/autodrive/Visual-RFT-main/share_data/lisa_val_sft.json")
    # ... 原有数据ini处理逻辑（make_conversation_image等） ...
    dataset_test = dataset_val_ini.map(make_conversation_image)
    

    # # 2. 分割验证集（原有逻辑不变）
    # if script_args.dataset_test_split:
    #     val_dataset = dataset[script_args.dataset_test_split]
    # else:
    #     dataset_split = dataset.train_test_split(test_size=0.1, seed=42)
    #     dataset = dataset_split["train"]
    #     val_dataset = dataset_split["test"]

    # 3. 初始化自定义回调（核心修改）


#    ###我修改的
#     print(11111111111)
#     eval_dataset = dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None
#     print(training_args.eval_strategy)
#     print(eval_dataset.shape)
#     print(datatest.shape)
   
    eval_callback = EvalEvery50EpochsCallback(eval_interval=1)
    trainer_cls = Qwen2VLGRPOTrainer if not training_args.use_vllm else Qwen2VLGRPOVLLMTrainer

    # Initialize the GRPO trainer
    trainer = trainer_cls(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=dataset_val,
        
        #eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        peft_config=get_peft_config(model_args),
        attn_implementation=model_args.attn_implementation,
        max_pixels=script_args.max_pixels,
        min_pixels=script_args.min_pixels#,
    #    callbacks=[eval_callback]  # 传入自定义回调
    
    )
    
    # 重要的是：在将回调添加到 Trainer 之前，将 Trainer 实例传递给回调
    # eval_callback.trainer = trainer 

    # # 将回调添加到 Trainer
    # trainer.add_callback(eval_callback)

    # Train and push the model to the Hub
    trainer.train()
    

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
   # print("展示一下是否用vllm",training_args.use_vllm ) #flase
    main(script_args, training_args, model_args)
