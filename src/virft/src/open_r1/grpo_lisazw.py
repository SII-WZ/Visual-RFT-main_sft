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
#我写的验证

class ValidationEvery50ItersCallback(TrainerCallback):
    """
    适配Qwen2VLGRPOTrainer的自定义回调，每50步执行验证，日志字段与训练完全对齐
    复用训练器内部的ref_model、奖励函数、KL计算逻辑，保证一致性
    """
    def __init__(
        self,
        val_dataset: Dataset,
        reward_funcs_names: List[str],
        log_save_path: str,
        max_pixels: int,
        min_pixels: int,
        eval_every: int = 50
    ):
        self.val_dataset = val_dataset
        self.reward_funcs_names = reward_funcs_names  # ["accuracy", "format"]
        self.log_save_path = log_save_path
        self.max_pixels = max_pixels
        self.min_pixels = min_pixels
        self.eval_every = eval_every

        # 确保日志目录存在
        os.makedirs(os.path.dirname(self.log_save_path), exist_ok=True)

    def on_step_end(
        self,
        args,
        state: TrainerState,
        control: TrainerControl,
        **kwargs
    ):
        """
        训练步骤结束后触发，每50步执行一次验证
        """
        if state.is_training and state.global_step % self.eval_every == 0 and state.global_step > 0:
            print(f"\n===== 开始执行第 {state.global_step} 步后的验证集评估 =====")
            # 获取训练器实例（kwargs["self"]对应Qwen2VLGRPOTrainer）
            trainer = kwargs.get("self")
            if trainer is None:
                print("警告：无法获取训练器实例，跳过本次验证")
                return control
            
            # 执行验证并保存结果
            self._run_validation(trainer, state.global_step)
            print(f"===== 第 {state.global_step} 步后的验证集评估完成，结果已保存至 {self.log_save_path} =====")
        
        return control

    def _run_validation(self, trainer: "Qwen2VLGRPOTrainer", current_step: int):
        """
        核心：复用Qwen2VLGRPOTrainer内部逻辑，计算所有对齐训练日志的指标
        """
        # 初始化验证结果，严格对齐训练日志字段
        val_results = {
            "kl": 0.0,
            "learning_rate": trainer.args.learning_rate,
            "loss": 0.0,  # 验证集不计算训练损失，保持格式对齐（如需真实验证损失可扩展）
            "reward": 0.0,
            "reward_std": 0.8923546671867371,
            **{f"rewards/{func_name}_reward": 0.0 for func_name in self.reward_funcs_names},
            "step": current_step,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        try:
            # -------------------------- 步骤1：生成验证集completions（复用训练器generate方法） --------------------------
            completions = trainer.generate(
                self.val_dataset,
                max_pixels=self.max_pixels,
                min_pixels=self.min_pixels,
                do_sample=False,
                temperature=0.0
            )

            # -------------------------- 步骤2：处理输入数据（与训练器compute_loss保持一致） --------------------------
            # 提取prompt和image
            prompts = [x["prompt"] for x in self.val_dataset]
            images = [x["image"] for x in self.val_dataset] if "image" in self.val_dataset.column_names else [None]*len(self.val_dataset)
            
            # 应用chat template并构建输入
            prompts_text = [trainer.maybe_apply_chat_template(example, trainer.processing_class)["prompt"] for example in self.val_dataset]
            prompt_inputs = trainer.processing_class(
                text=prompts_text,
                images=images,
                return_tensors="pt",
                padding=True,
                padding_side="left",
                add_special_tokens=False,
            )
            prompt_inputs = trainer._prepare_inputs(prompt_inputs)

            # -------------------------- 步骤3：计算真实KL散度（复用训练器_ref_model和_get_per_token_logps） --------------------------
            # 生成prompt_completion_ids（用于KL计算）
            with trainer.unwrap_model_for_generation(trainer.model, trainer.accelerator) as unwrapped_model:
                prompt_completion_ids = unwrapped_model.generate(
                    **prompt_inputs,
                    generation_config=trainer.generation_config
                )

            # 提取prompt长度、completion_ids和mask（与训练器逻辑一致）
            prompt_length = prompt_inputs["input_ids"].size(1)
            completion_ids = prompt_completion_ids[:, prompt_length:]
            prompt_mask = prompt_inputs["attention_mask"].repeat_interleave(trainer.num_generations, dim=0)

            # 构建completion_mask（屏蔽eos之后的token）
            is_eos = completion_ids == trainer.processing_class.eos_token_id
            device = trainer.accelerator.device
            eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
            eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
            sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
            completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

            # 拼接attention mask
            attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
            pixel_values = prompt_inputs["pixel_values"].repeat(trainer.num_generations, 1) if "pixel_values" in prompt_inputs else None
            image_grid_thw = prompt_inputs["image_grid_thw"].repeat_interleave(trainer.num_generations, dim=0) if "image_grid_thw" in prompt_inputs else None

            # 计算当前模型和参考模型的per-token logps
            per_token_logps = trainer._get_per_token_logps(
                trainer.model, prompt_completion_ids, attention_mask, pixel_values, image_grid_thw
            )
            per_token_logps = per_token_logps[:, prompt_length - 1 :]

            # 计算参考模型logps（复用训练器的ref_model）
            with torch.inference_mode():
                if trainer.ref_model is not None:
                    ref_per_token_logps = trainer._get_per_token_logps(
                        trainer.ref_model, prompt_completion_ids, attention_mask, pixel_values, image_grid_thw
                    )
                else:
                    with trainer.accelerator.unwrap_model(trainer.model).disable_adapter():
                        ref_per_token_logps = trainer._get_per_token_logps(
                            trainer.model, prompt_completion_ids, attention_mask, pixel_values, image_grid_thw
                        )
            ref_per_token_logps = ref_per_token_logps[:, prompt_length - 1 :]

            # 计算KL散度（与训练器compute_loss完全一致）
            per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
            mean_kl = ((per_token_kl * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
            val_results["kl"] = float(trainer.accelerator.gather_for_metrics(mean_kl).mean().item())

            # -------------------------- 步骤4：计算奖励指标（复用训练器reward_funcs） --------------------------
            # 重复prompt以匹配生成数量
            prompts_repeated = [prompt for prompt in prompts for _ in range(trainer.num_generations)]
            completions_repeated = completions

            # 计算各奖励函数的结果
            rewards_per_func = torch.zeros(len(prompts_repeated), len(trainer.reward_funcs), device=device)
            for i, (reward_func, reward_processing_class) in enumerate(
                zip(trainer.reward_funcs, trainer.reward_processing_classes)
            ):
                if isinstance(reward_func, torch.nn.Module):
                    # 复用训练器的对话模板和输入处理逻辑
                    if trainer.is_conversational(self.val_dataset[0]):
                        messages = [{"messages": p + c} for p, c in zip(prompts_repeated, completions_repeated)]
                        texts = [trainer.apply_chat_template(x, reward_processing_class)["text"] for x in messages]
                    else:
                        texts = [p + c for p, c in zip(prompts_repeated, completions_repeated)]
                    
                    reward_inputs = reward_processing_class(
                        texts, return_tensors="pt", padding=True, padding_side="right", add_special_tokens=False
                    )
                    reward_inputs = trainer._prepare_inputs(reward_inputs)
                    
                    with torch.inference_mode():
                        rewards_per_func[:, i] = reward_func(**reward_inputs).logits[:, 0]
                else:
                    # 自定义奖励函数（accuracy/format）
                    reward_kwargs = {key: [] for key in self.val_dataset[0].keys() if key not in ["prompt", "completion"]}
                    for key in reward_kwargs:
                        for example in self.val_dataset:
                            reward_kwargs[key].extend([example[key]] * trainer.num_generations)
                    
                    output_reward_func = reward_func(prompts=prompts_repeated, completions=completions_repeated, **reward_kwargs)
                    rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)

            # -------------------------- 步骤5：填充奖励指标（对齐训练日志） --------------------------
            # 整体奖励均值
            total_rewards = rewards_per_func.sum(dim=1)
            val_results["reward"] = float(trainer.accelerator.gather_for_metrics(total_rewards).mean().item())

            # 各子奖励均值（对应rewards/accuracy_reward、rewards/format_reward）
            reward_per_func_mean = trainer.accelerator.gather_for_metrics(rewards_per_func).mean(0)
            for i, func_name in enumerate(self.reward_funcs_names):
                val_results[f"rewards/{func_name}_reward"] = float(reward_per_func_mean[i].item())

            # 奖励标准差（与训练日志对齐）
            std_grouped_rewards = total_rewards.view(-1, trainer.num_generations).std(dim=1)
            val_results["reward_std"] = float(trainer.accelerator.gather_for_metrics(std_grouped_rewards).mean().item())

        except Exception as e:
            print(f"验证集评估过程中出现错误：{e}")
        finally:
            # 保存验证结果
            self._save_validation_results(val_results)

    def _save_validation_results(self, val_results: Dict):
        """
        追加写入验证结果，保持与训练日志一致的格式
        """
        with open(self.log_save_path, "a", encoding="utf-8") as f:
            json.dump(val_results, f, ensure_ascii=False)
            f.write("\n" + "="*120 + "\n")
        
        # 保存最新结果（方便快速查看）
        latest_log_path = self.log_save_path.replace(".json", "_latest.json")
        with open(latest_log_path, "w", encoding="utf-8") as f:
            json.dump(val_results, f, ensure_ascii=False, indent=2)


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

    dataset = Dataset.from_json("/inspire/hdd/global_user/zhouwei-240108540164/autodrive/Visual-RFT-main/share_data/lisa_train_sft.json")

#    Format into conversation
    def make_conversation(example):
        return {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": example["problem"]},
            ],
        }

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
        return {"image": PIL.Image.open(image_path), "prompt": formatted_conversation, 'solution': res}

    dataset = dataset.map(make_conversation_image)
    
   ###我修改的
    reward_funcs_names = script_args.reward_funcs  # ["accuracy", "format"]
    reward_funcs = [reward_funcs_registry[func] for func in reward_funcs_names]
    
    dataset_val = Dataset.from_json("/inspire/hdd/global_user/zhouwei-240108540164/autodrive/Visual-RFT-main/share_data/lisa_val_sft.json")
    # ... 原有数据处理逻辑（make_conversation_image等） ...
    dataset_val = dataset_val.map(make_conversation_image)

    # # 2. 分割验证集（原有逻辑不变）
    # if script_args.dataset_test_split:
    #     val_dataset = dataset[script_args.dataset_test_split]
    # else:
    #     dataset_split = dataset.train_test_split(test_size=0.1, seed=42)
    #     dataset = dataset_split["train"]
    #     val_dataset = dataset_split["test"]

    # 3. 初始化自定义回调（核心修改）
    val_callback = ValidationEvery50ItersCallback(
        val_dataset=dataset_val,
        reward_funcs_names=reward_funcs_names,
        log_save_path="./out/lisa_train_GIoU/grpo_validation_log.json",
        max_pixels=script_args.max_pixels,
        min_pixels=script_args.min_pixels,
        eval_every=1
    )

   ###我修改的
    
    
    trainer_cls = Qwen2VLGRPOTrainer if not training_args.use_vllm else Qwen2VLGRPOVLLMTrainer

    # Initialize the GRPO trainer
    trainer = trainer_cls(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        peft_config=get_peft_config(model_args),
        attn_implementation=model_args.attn_implementation,
        max_pixels=script_args.max_pixels,
        min_pixels=script_args.min_pixels,
         callbacks=[val_callback]  # 传入自定义回调
    )

    # Train and push the model to the Hub
    trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


if __name__ == "__main__":
  #  1. 构造脚本参数（GRPOScriptArguments）
    # ======================================
    script_args = GRPOScriptArguments(
        reward_funcs=["accuracy", "format"],
        max_pixels=401408,  # 对应torchrun命令中的--max_pixels
        min_pixels=3136,
     #   val_log_path="./out/lisa_train_GIoU/grpo_validation_log.json",
        dataset_name="NOT_USED"
    )

    # ======================================
    # 2. 构造训练参数（GRPOConfig）
    # 对应torchrun命令中的所有训练相关参数
    # ======================================
    training_args = GRPOConfig(
        # 基础配置
        output_dir="out/lisa_train_GIoU",  # 对应--output_dir
        run_name="Qwen2-VL-2B-GRPO-groud_lisa_train",  # 对应--run_name
        push_to_hub=False,  # 命令中未指定开启，默认False
        num_train_epochs=6,  # 对应--num_train_epochs
        save_steps=150,  # 对应--save_steps
        save_only_model=True,  # 对应--save_only_model
        logging_steps=10,  # 对应--logging_steps
        
        # 批次配置
        per_device_train_batch_size=2,  # 对应--per_device_train_batch_size
        gradient_accumulation_steps=2,  # 对应--gradient_accumulation_steps
        
        # 精度配置
        # dtype="float16",  # 对应--dtype
        bf16=True,  # 对应--bf16 "true"
        
        # 模型配置
        gradient_checkpointing=True,  # 对应--gradient_checkpointing true
        num_generations=2,  # 对应--num_generations 4
        
        # 分布式/加速配置
        deepspeed="/inspire/hdd/global_user/zhouwei-240108540164/autodrive/Visual-RFT-main/src/virft/local_scripts/zero3.json",  # 对应--deepspeed
        eval_strategy="no",  # 关闭默认评估，使用自定义回调
        use_vllm=False,  # 默认不使用VLLM
        
        # 其他默认配置（保持TRL默认值即可）
        remove_unused_columns=False,
        fp16=False,  # 与bf16二选一，命令中指定bf16
        report_to="none"
    )

    # ======================================
    # 3. 构造模型参数（ModelConfig）
    # 对应torchrun命令中的模型相关参数
    # ======================================
    model_args = ModelConfig(
        model_name_or_path="/inspire/hdd/global_user/zhouwei-240108540164/autodrive/Visual-RFT-main/Qwen/Qwen2-VL-2B-Instruct",  # 对应--model_name_or_path
        attn_implementation="flash_attention_2"   # 对应--attn_implementation

    )

    # ======================================
    # 4. 启动训练
    # ======================================
    main(script_args, training_args, model_args)
