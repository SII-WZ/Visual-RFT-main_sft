import torch
import json
import os
import re
import math
from datetime import datetime
from typing import Optional, List, Dict
from PIL import Image
from tqdm import tqdm
import logging
from datasets import Dataset

# 导入transformers相关依赖（确保和你的环境版本兼容）
from transformers import (
    Qwen2VLForConditionalGeneration,
    AutoProcessor,
    __version__ as transformers_version
)
from qwen_vl_utils import process_vision_info

# ---------------------- 全局配置（根据你的本地路径确认，无需额外修改）----------------------
CONFIG = {
    "main_model_path": "./out/lisa_train_GIoU_7b/checkpoint-50",  # 训练好的主模型本地路径
    "ref_model_path": "./Qwen/Qwen2-VL-7B-Instruct",            # 参考模型本地路径
    "data_path": "/inspire/hdd/global_user/zhouwei-240108540164/autodrive/Visual-RFT-main/share_data/lisa_val_sft.json",  # 你的JSON数据路径lisa_train_sft_test.json" lisa_val_sft.json
    "save_dir": "./lisa_evaluation",                     # 指标保存目录
    "num_generations": 4,                                        # 每个样本生成的补全数量
    "beta": 0.1,                                                 # GRPO中的beta（KL损失权重）
    "max_prompt_length": 1024,                                   # 最大prompt长度
    "max_new_tokens": 128,                                       # 生成的最大新token数
    "torch_dtype": torch.bfloat16,                               # 模型数据类型
}

# ---------------------- 固定模板（和你的训练逻辑对齐）----------------------
# QUESTION_TEMPLATE = "{Question} Output the thinking process in  and your grouding box. Following \"\n<answer>(x1,y1),(x2,y2)</answer>)\" format."
QUESTION_TEMPLATE = "{Question} Output the thinking process in <think> </think> and your grouding box. Following \"<think> thinking process </think>\n<answer>(x1,y1),(x2,y2)</answer>)\" format."
# ---------------------- 日志配置 ----------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# ---------------------- 核心工具函数（完全对齐训练逻辑，新增GIoU计算）----------------------
def process_image(image_path: str) -> Image.Image:
    """加载图片（兼容绝对路径/相对路径）"""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"图片不存在: {image_path}")
    return Image.open(image_path)

def compute_giou(gt_bbox, student_bbox):
    """
    计算GIoU（训练侧同款），并做+1缩放，返回范围[0,2]
    gt_bbox/student_bbox 格式：[(x1,y1), (x2,y2)]（已归一化到0-1）
    """
    try:
        # 提取坐标
        x1_gt, y1_gt = gt_bbox[0]
        x2_gt, y2_gt = gt_bbox[1]
        
        x1_st, y1_st = student_bbox[0]
        x2_st, y2_st = student_bbox[1]

        # 计算交集
        x1_inter = max(x1_gt, x1_st)
        y1_inter = max(y1_gt, y1_st)
        x2_inter = min(x2_gt, x2_st)
        y2_inter = min(y2_gt, y2_st)

        inter_width = max(0, x2_inter - x1_inter)
        inter_height = max(0, y2_inter - y1_inter)
        inter_area = inter_width * inter_height

        # 计算并集
        gt_area = (x2_gt - x1_gt) * (y2_gt - y1_gt)
        student_area = (x2_st - x1_st) * (y2_st - y1_st)
        union_area = gt_area + student_area - inter_area

        # 计算IoU
        iou = inter_area / union_area if union_area > 0 else 0.0

        # 计算最小包围盒
        x1_c = min(x1_gt, x1_st)
        y1_c = min(y1_gt, y1_st)
        x2_c = max(x2_gt, x2_st)
        y2_c = max(y2_gt, y2_st)
        c_area = (x2_c - x1_c) * (y2_c - y1_c)

        # 计算GIoU并缩放（范围[-1,1] -> [0,2]）
        giou = iou - (c_area - union_area) / c_area if c_area > 0 else iou
        giou_scaled = giou + 1.0  # 和训练侧完全对齐，缩放后范围[0,2]
        
        return giou_scaled
    except Exception as e:
        logger.debug(f"计算GIoU失败: {e}")
        return 0.0

def extract_bbox_coords(text: str) -> Optional[List[int]]:
    """
    从文本中提取bbox原始坐标（和训练侧正则对齐），返回[x1,y1,x2,y2]（未归一化）
    """
    try:
        pattern = r"\(\s*(\d+)\s*,\s*(\d+)\s*\)\s*,\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)"
        matches = re.findall(pattern, text)
        if not matches:
            return None
        x1, y1, x2, y2 = map(int, matches[0])
        # 校验坐标有效性（非负，且x1<x2, y1<y2）
        if x1 >= 0 and y1 >= 0 and x2 > x1 and y2 > y1 and x2 <= 1000 and y2 <= 1000:
            return [x1, y1, x2, y2]
        else:
            return None
    except Exception as e:
        logger.debug(f"提取bbox坐标失败: {e}")
        return None

def extract_normalized_bbox(text: str) -> Optional[List[List[float]]]:
    """
    提取归一化后的bbox（和训练侧对齐，除以1000），返回格式[(x1,y1), (x2,y2)]
    """
    coords = extract_bbox_coords(text)
    if not coords:
        return None
    x1, y1, x2, y2 = coords
    # 固定除以1000归一化（和训练侧完全对齐，不使用图片实际尺寸）
    norm_coords = [val / 1000.0 for val in coords]
    return [(norm_coords[0], norm_coords[1]), (norm_coords[2], norm_coords[3])]

def extract_answer_content(text: str, tag: str = "answer") -> Optional[str]:
    """
    从<answer>标签中提取内容（和训练侧对齐）
    """
    try:
        pattern = rf"<{tag}>(.*?)</{tag}>"
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()
        else:
            return None
    except Exception as e:
        logger.debug(f"提取<{tag}>标签内容失败: {e}")
        return None

def apply_chat_template(messages: List[Dict], processor: AutoProcessor) -> str:
    """应用chat模板生成prompt文本"""
    return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

def compute_format_reward(completion: str) -> float:
    """
    计算格式奖励（和训练侧完全对齐）：
    要求同时包含「」和「<answer>...</answer>」，且完整匹配格式
    符合返回1.0，不符合返回0.0
    """
    try:
        pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
        match = re.fullmatch(pattern, completion.strip(), re.DOTALL)
        return 1.0 if match else 0.0
    except Exception as e:
        logger.debug(f"格式奖励计算失败: {e}")
        return 0.0


def compute_accuracy_reward(completion: str, solution: str) -> float:
    """
    计算精度奖励（和训练侧完全对齐）：
    1.  从completion和solution中提取<answer>标签内容
    2.  提取bbox并计算GIoU（+1缩放）
    3.  完全字符串匹配返回2.0（兜底最高奖励）
    """
    try:
        # 步骤1：提取<answer>标签内容
        completion_answer = extract_answer_content(completion)
        solution_answer = extract_answer_content(solution)
        if not completion_answer or not solution_answer:
            return 0.0
        
        # 步骤2：兜底：完全字符串匹配返回2.0（和训练侧对齐）
        if completion_answer == solution_answer:
            return 2.0
        
        # 步骤3：提取归一化bbox并计算GIoU
        student_bbox = extract_normalized_bbox(completion_answer)
        gt_bbox = extract_normalized_bbox(solution_answer)
        if not student_bbox or not gt_bbox:
            return 0.0
        
        # 步骤4：计算GIoU（已缩放，范围[0,2]）
        return compute_giou(gt_bbox, student_bbox)
    except Exception as e:
        logger.debug(f"精度奖励计算失败: {e}")
        return 0.0

def compute_combined_rewards(completion: str, example: Dict) -> Dict[str, float]:
    """
    计算组合奖励（和训练侧完全对齐：accuracy_reward + format_reward）
    返回细分奖励和总奖励
    """
    try:
        solution = example["solution"]
        
        # 计算精度奖励（GIoU，范围[0,2]）
        accuracy_reward = compute_accuracy_reward(completion, solution)
        
        # 计算格式奖励（严格格式校验，范围[0,1]）
        format_reward = compute_format_reward(completion)
        
        # 总奖励（和训练侧一致：直接求和，范围[0,3]）
        total_reward = accuracy_reward + format_reward
        
        return {
            "accuracy_reward": accuracy_reward,
            "format_reward": format_reward,
            "total_reward": total_reward
        }
    except Exception as e:
        logger.debug(f"组合奖励计算失败: {e}")
        return {"accuracy_reward": 0.0, "format_reward": 0.0, "total_reward": 0.0}

# ---------------------- 你的数据处理逻辑（make_conversation_image，完整保留并优化）----------------------
def make_conversation_image(example):
    """处理单条样本，提取图片、格式化对话、生成solution（和训练逻辑完全对齐）"""
    conversations = example["conversations"]
    formatted_conversation = []
    image_path = ""
    res = ""

    for message in conversations:
        if message["from"] == "user":
            # 检查是否包含图片标签
            if "<img>" in message["value"] and "</img>" in message["value"]:
                # 提取图片路径
                image_path = message["value"].split("<img>")[1].split("</img>")[0].strip()
                # 提取纯文本内容并格式化
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
            # 生成带<answer>标签的solution（和训练侧对齐）
            res = '<answer> ' + message["value"] + ' </answer>'
    
    # 加载图片（确保路径有效）
    if image_path and os.path.exists(image_path):
        image = Image.open(image_path)
    else:
        raise FileNotFoundError(f"无效的图片路径: {image_path}")
    
    return {"image": image, "prompt": formatted_conversation, "solution": res, "image_path": image_path}

# ---------------------- GRPO核心指标计算类（完全对齐训练奖励逻辑）----------------------
class GRPOStandaloneEvaluator:
    def __init__(self, config: Dict):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # 新增 accuracy_reward、format_reward 指标，和训练逻辑对齐
        self.metrics = {
            "loss": [],
            "kl": [],
            "reward": [],
            "reward_std": [],
            "accuracy_reward": [],
            "format_reward": [],
            "completion_length": [],
        }

        # 1. 加载processor
        self.processor = self._load_processor()
        # 2. 加载主模型和参考模型
        self.main_model = self._load_model(config["main_model_path"])
        self.ref_model = self._load_model(config["ref_model_path"])
        # 3. 初始化保存目录
        os.makedirs(config["save_dir"], exist_ok=True)
        # 4. 预处理数据集（使用你的make_conversation_image逻辑）
        self.dataset = self._preprocess_dataset()

    def _load_processor(self) -> AutoProcessor:
        """加载处理器（优先从主模型路径加载）"""
        try:
            processor = AutoProcessor.from_pretrained(
                self.config["main_model_path"],
                local_files_only=True
            )
            logger.info("处理器加载成功（从主模型路径）")
            return processor
        except Exception as e:
            logger.warning(f"主模型路径加载处理器失败，尝试从参考模型路径: {e}")
            processor = AutoProcessor.from_pretrained(
                self.config["ref_model_path"],
                local_files_only=True
            )
            logger.info("处理器加载成功（从参考模型路径）")
            return processor

    def _load_model(self, model_path: str) -> Qwen2VLForConditionalGeneration:
        """加载Qwen2-VL模型（本地加载）"""
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=self.config["torch_dtype"],
            device_map="auto" if self.device == "cuda" else "cpu",
            local_files_only=True,
            attn_implementation="flash_attention_2" if self.device == "cuda" else None
        ).eval()
        logger.info(f"模型加载成功: {model_path}")
        return model

    def _preprocess_dataset(self):
        """预处理数据集（使用你的make_conversation_image逻辑，生成适配测评的数据集）"""
        # 从JSON加载数据集
        dataset_ini = Dataset.from_json(self.config["data_path"])
        logger.info(f"原始数据集加载成功，共 {len(dataset_ini)} 条样本")

        # 应用你的数据处理逻辑
        dataset = dataset_ini.map(make_conversation_image)
        logger.info(f"数据集预处理完成，共 {len(dataset)} 条有效样本")

        return dataset

    def _prepare_inputs(self, example: Dict) -> Dict:
        """准备模型输入（基于预处理后的样本，和训练逻辑对齐）"""
        formatted_conversation = example["prompt"]
        image = example["image"]
        image_path = example["image_path"]

        # 补充图片内容到对话中（适配Qwen2-VL输入格式）
        for msg in formatted_conversation:
            if msg["role"] == "user" and isinstance(msg["content"], list):
                for content_item in msg["content"]:
                    if content_item["type"] == "image":
                        content_item["image"] = image_path

        # 处理文本和图片
        text = apply_chat_template(formatted_conversation, self.processor)
        image_inputs, _ = process_vision_info(formatted_conversation)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            padding=True,
            return_tensors="pt",
            padding_side="left"
        )

        # 截断prompt（如果超过最大长度）
        if self.config["max_prompt_length"] is not None:
            inputs["input_ids"] = inputs["input_ids"][:, -self.config["max_prompt_length"]:]
            inputs["attention_mask"] = inputs["attention_mask"][:, -self.config["max_prompt_length"]:]

        return {k: v.to(self.device) for k, v in inputs.items()}

    def _get_per_token_logps(self, model: Qwen2VLForConditionalGeneration, input_ids: torch.Tensor, 
                             attention_mask: torch.Tensor, pixel_values: torch.Tensor, 
                             image_grid_thw: torch.Tensor) -> torch.Tensor:
        """计算每个token的对数概率"""
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                return_dict=True
            )
        
        logits = outputs.logits[:, :-1, :]
        input_ids_shifted = input_ids[:, 1:]

        # 计算log_softmax
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        # 提取对应token的log概率
        per_token_logps = torch.gather(log_probs, 2, input_ids_shifted.unsqueeze(2)).squeeze(2)

        # 用attention mask掩码无效token
        attention_mask_shifted = attention_mask[:, 1:]
        per_token_logps = per_token_logps * attention_mask_shifted

        return per_token_logps

    def _compute_grpo_metrics(self, example: Dict) -> Dict:
        """计算单个样本的GRPO核心指标（loss、KL等，包含多维度奖励）"""
        num_generations = self.config["num_generations"]
        beta = self.config["beta"]

        # 1. 准备输入
        inputs = self._prepare_inputs(example)
        prompt_length = inputs["input_ids"].size(1)

        # 2. 生成补全结果（重复num_generations次）
        inputs_repeated = {
            k: v.repeat(num_generations, 1) if v.dim() == 2 else v.repeat(num_generations, 1, 1, 1)
            for k, v in inputs.items()
        }

        with torch.no_grad():
            generated_ids = self.main_model.generate(
                **inputs_repeated,
                max_new_tokens=self.config["max_new_tokens"],
                do_sample=False
            )

        # 3. 拆分prompt和completion
        prompt_ids = generated_ids[:, :prompt_length]
        completion_ids = generated_ids[:, prompt_length:]
        prompt_mask = inputs["attention_mask"].repeat(num_generations, 1)

        # 4. 生成completion mask（掩码EOS之后的token）
        eos_token_id = self.processor.tokenizer.eos_token_id
        is_eos = completion_ids == eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=self.device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=self.device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

        # 5. 拼接attention mask
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        pixel_values = inputs["pixel_values"].repeat(num_generations, 1, 1, 1)
        image_grid_thw = inputs.get("image_grid_thw", torch.tensor([])).repeat_interleave(num_generations, dim=0)

        # 6. 计算per token logps（主模型和参考模型）
        main_logps = self._get_per_token_logps(
            self.main_model, generated_ids, attention_mask, pixel_values, image_grid_thw
        )[:, prompt_length - 1:]

        ref_logps = self._get_per_token_logps(
            self.ref_model, generated_ids, attention_mask, pixel_values, image_grid_thw
        )[:, prompt_length - 1:]

        # 7. 计算KL散度
        per_token_kl = torch.exp(ref_logps - main_logps) - (ref_logps - main_logps) - 1

        # 8. 解码completion并计算多维度奖励（和训练侧对齐）
        completions = self.processor.batch_decode(completion_ids, skip_special_tokens=True)
        rewards = []
        accuracy_rewards = []
        format_rewards = []
        for completion in completions:
            reward_dict = compute_combined_rewards(completion, example)
            rewards.append(reward_dict["total_reward"])
            accuracy_rewards.append(reward_dict["accuracy_reward"])
            format_rewards.append(reward_dict["format_reward"])

        # 转换为张量用于后续计算
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)

        # 计算细分奖励的样本平均值
        sample_accuracy_reward = torch.tensor(accuracy_rewards, dtype=torch.float32).mean().item()
        sample_format_reward = torch.tensor(format_rewards, dtype=torch.float32).mean().item()

        # 9. 计算优势（Advantage）
        mean_reward = rewards.mean()
        std_reward = rewards.std()
        advantages = (rewards - mean_reward) / (std_reward + 1e-4)

        # 10. 计算GRPO损失
        per_token_loss = torch.exp(main_logps - main_logps.detach()) * advantages.unsqueeze(1)
        per_token_loss = -(per_token_loss - beta * per_token_kl)

        # 11. 聚合指标
        completion_length = completion_mask.sum(dim=1).float().mean().item()
        loss = ((per_token_loss * completion_mask).sum(dim=1) / (completion_mask.sum(dim=1) + 1e-4)).mean().item()
        mean_kl = ((per_token_kl * completion_mask).sum(dim=1) / (completion_mask.sum(dim=1) + 1e-4)).mean().item()

        return {
            "loss": loss,
            "kl": mean_kl,
            "reward": rewards.mean().item(),
            "reward_std": std_reward.item(),
            "accuracy_reward": sample_accuracy_reward,
            "format_reward": sample_format_reward,
            "completion_length": completion_length,
            "completions": completions
        }

    def _save_metrics_to_local(self, final_metrics: Dict, point=50):
        """将最终指标保存到本地（JSON + 完整归档，解决dtype无法序列化问题）"""
        save_dir = self.config["save_dir"]
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 关键修改1：复制final_metrics并转换不可序列化的对象（torch.dtype -> 字符串）
        serializable_metrics = final_metrics.copy()
        # 处理config中的torch_dtype（转换为字符串）
        if "config" in serializable_metrics:
            config = serializable_metrics["config"].copy()
            if "torch_dtype" in config:
                # 将torch.bfloat16/torch.float32转换为字符串
                config["torch_dtype"] = str(config["torch_dtype"]).split(".")[-1]  # 提取 dtype 名称（如 bfloat16）
            serializable_metrics["config"] = config

        # 1. 保存完整指标（JSON，包含配置和详细结果，使用序列化后的指标）
        json_path = os.path.join(save_dir, f"7b_grpo_metrics_{point}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(serializable_metrics, f, indent=4, ensure_ascii=False)  # 写入序列化后的指标
        logger.info(f"完整指标已保存到: {json_path}")

    def run_evaluation(self,point_num=50):
        """运行完整评估流程（可直接执行）"""
        logger.info("开始执行GRPO指标评估...")

        # 1. 逐样本计算指标
        for example in tqdm(self.dataset, desc="计算GRPO指标"):
            try:
                sample_metrics = self._compute_grpo_metrics(example)
                
                # 更新全局指标
                for key in self.metrics.keys():
                    self.metrics[key].append(sample_metrics[key])
            except Exception as e:
                image_path = example.get("image_path", "未知路径")
                logger.error(f"处理样本 {image_path} 失败: {e}")
                continue

        # 2. 计算最终平均指标
        final_metrics = {
            f"average_{key}": sum(val) / len(val) if len(val) > 0 else 0.0
            for key, val in self.metrics.items()
        }
        final_metrics["sample_count"] = len(self.metrics["loss"])
        final_metrics["config"] = self.config

        # 3. 打印最终指标
        logger.info("="*50)
        logger.info("GRPO 评估最终结果")
        logger.info("="*50)
        for key, val in final_metrics.items():
            if not key.startswith("config") and key != "sample_count":
                logger.info(f"{key}: {val:.8f}")
        # 额外打印细分奖励（和训练格式对齐）
        logger.info(f"rewards/accuracy_reward: {final_metrics.get('average_accuracy_reward', 0.0):.8f}")
        logger.info(f"rewards/format_reward: {final_metrics.get('average_format_reward', 0.0):.8f}")
        logger.info(f"有效样本数: {final_metrics['sample_count']}")

        # 4. 保存指标到本地
        self._save_metrics_to_local(final_metrics, point=point_num)

# ---------------------- 主函数（直接运行入口）----------------------
if __name__ == "__main__":
    try:
        # # 初始化评估器
        # evaluator = GRPOStandaloneEvaluator(CONFIG)
        # # 运行完整评估
        # evaluator.run_evaluation(point_num=50)
        
        # CONFIG["main_model_path"] = "./out/lisa_train_GIoU_2b/checkpoint-100"
        # evaluator = GRPOStandaloneEvaluator(CONFIG)
        # evaluator.run_evaluation(point_num=100)
        
        # CONFIG["main_model_path"] = "./out/lisa_train_GIoU_2b/checkpoint-150"
        # evaluator = GRPOStandaloneEvaluator(CONFIG)
        # evaluator.run_evaluation(point_num=150)
        
        # CONFIG["main_model_path"] = "./out/lisa_train_GIoU_2b/checkpoint-200"
        # evaluator = GRPOStandaloneEvaluator(CONFIG)
        # evaluator.run_evaluation(point_num=200)
        
        CONFIG["main_model_path"] = "./out/lisa_train_GIoU_7b/checkpoint-250"
        evaluator = GRPOStandaloneEvaluator(CONFIG)
        evaluator.run_evaluation(point_num=250)
        # CONFIG["main_model_path"] = "./out/lisa_train_GIoU_2b/checkpoint-300"
        # evaluator = GRPOStandaloneEvaluator(CONFIG)
        # evaluator.run_evaluation(point_num=300)
        
        CONFIG["main_model_path"] = "./out/lisa_train_GIoU_7b/checkpoint-350"
        evaluator = GRPOStandaloneEvaluator(CONFIG)
        evaluator.run_evaluation(point_num=350)
        
        # CONFIG["main_model_path"] = "./out/lisa_train_GIoU_2b/checkpoint-400"
        # evaluator = GRPOStandaloneEvaluator(CONFIG)
        # evaluator.run_evaluation(point_num=400)
        
        CONFIG["main_model_path"] = "./out/lisa_train_GIoU_7b/checkpoint-450"
        evaluator = GRPOStandaloneEvaluator(CONFIG)
        evaluator.run_evaluation(point_num=450)
        
        
        
        
        logger.info("GRPO 评估流程全部完成！结果已保存到 ./lisa_evaluation 目录")
    except Exception as e:
        logger.error(f"评估流程异常终止: {e}")
        raise