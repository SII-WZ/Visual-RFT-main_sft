import torch
import json
import os
import re
from PIL import Image
from tqdm import tqdm
import logging

# 你的依赖
from transformers import (
    Qwen2VLForConditionalGeneration,
    AutoProcessor,
)
from qwen_vl_utils import process_vision_info

# ---------------------- 路径配置（只改这里！）----------------------
CONFIG = {
    "main_model_path": "./out/Qwen2-VL-2B-Instruct_GRPO_coco_base65cate_6k/checkpoint-1500",
    "ref_model_path": "./Qwen/Qwen2-VL-2B-Instruct",
    "coco_annotation": "/inspire/hdd/global_user/zhouwei-2025/autodrive/Visual-RFT-main/share_data/coco_val/annotations/instances_val2017.json",
    "coco_image_root": "/inspire/hdd/global_user/zhouwei-2025/autodrive/Visual-RFT-main/share_data/coco_val/images",
    "save_dir": "./coco_evaluation",
    "max_new_tokens": 256,
    "torch_dtype": torch.bfloat16,
}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------- COCO 65 类（你训练用的）----------------------
COCO65_NAMES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
    'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
    'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
    'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet',
    'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone'
]

# ---------------------- 提示词（和你训练完全一致）----------------------
QUESTION_TEMPLATE = """Detect all objects belonging to the category '{category}' in the image, and provide the bounding boxes (between 0 and 1000, integer) and confidence (between 0 and 1, with two decimal places).
If no object belonging to the category '{category}' in the image, return 'No Objects'.
Output the thinking process in   and final answer in <answer> </answer> tags.
The output answer format should be as follows:
 ... 
<answer>[{{'Position': [x1, y1, x2, y2], 'Confidence': number}}, ...]</answer>
Please strictly follow the format."""

# ---------------------- 工具函数（和你训练 1:1）----------------------
def extract_bbox(response):
    start_tag = "<answer>"
    end_tag = "</answer>"
    if start_tag not in response:
        return []
    s = response.find(start_tag) + len(start_tag)
    e = response.find(end_tag) if end_tag in response else len(response)
    content = response[s:e].strip()
    if not content.endswith("]"):
        content = content.rsplit("},", 1)[0] + "}]"
    content = content.replace("'", '"')
    try:
        return json.loads(content)
    except:
        return []

def compute_iou(a, b):
    x1, y1, x2, y2 = a
    x1_2, y1_2, x2_2, y2_2 = b
    xi1 = max(x1, x1_2)
    yi1 = max(y1, y1_2)
    xi2 = min(x2, x2_2)
    yi2 = min(y2, y2_2)
    if xi2 <= xi1 or yi2 <= yi1:
        return 0.0
    inter = (xi2 - xi1) * (yi2 - yi1)
    area_a = (x2 - x1) * (y2 - y1)
    area_b = (x2_2 - x1_2) * (y2_2 - y1_2)
    return inter / (area_a + area_b - inter) if (area_a + area_b - inter) > 0 else 0

# ---------------------- 加载 COCO 数据集（官方格式直接读）----------------------
def load_coco_as_eval_data(anno_path, img_root):
    with open(anno_path, 'r') as f:
        coco = json.load(f)
    cat_map = {cat['id']: cat['name'] for cat in coco['categories']}
    img_map = {img['id']: img['file_name'] for img in coco['images']}
    data = []
    for ann in coco['annotations']:
        cat = cat_map[ann['category_id']]
        if cat not in COCO65_NAMES:
            continue
        x, y, w, h = ann['bbox']
        x1, y1, x2, y2 = int(x), int(y), int(x+w), int(y+h)
        data.append({
            "image": os.path.join(img_root, img_map[ann['image_id']]),
            "category": cat,
            "gt": [{"Position": [x1, y1, x2, y2], "Confidence": 1.0}]
        })
    return data

# ---------------------- 评估器（极简、稳定、不报错）----------------------
class CocoGRPOEvaluator:
    def __init__(self, config):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = AutoProcessor.from_pretrained(config["ref_model_path"])
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            config["main_model_path"],
            torch_dtype=config["torch_dtype"],
            device_map="auto",
            attn_implementation="flash_attention_2"
        ).eval()
        self.data = load_coco_as_eval_data(config["coco_annotation"], config["coco_image_root"])
        os.makedirs(config["save_dir"], exist_ok=True)
        logger.info(f"Loaded {len(self.data)} COCO 65-class samples")

    @torch.no_grad()
    def run(self):
        total_iou = []
        total_format = []
        for sample in tqdm(self.data[:500]):  # 取前500张快速评估
            try:
                img_path = sample["image"]
                cat = sample["category"]
                gt = sample["gt"]

                messages = [
                    {"role": "user", "content": [
                        {"type": "image", "image": img_path},
                        {"type": "text", "text": QUESTION_TEMPLATE.format(category=cat)}
                    ]}
                ]

                inputs = self.processor(
                    text=[self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)],
                    images=[Image.open(img_path)],
                    return_tensors="pt"
                ).to(self.device)

                output = self.model.generate(
                    **inputs, max_new_tokens=self.config["max_new_tokens"], do_sample=False
                )
                response = self.processor.decode(output[0], skip_special_tokens=True)

                # 奖励
                pred = extract_bbox(response)
                format_ok = 1.0 if "<answer>" in response else 0.0
                iou = compute_iou(pred[0]["Position"], gt[0]["Position"]) if len(pred) > 0 else 0.0

                total_iou.append(iou)
                total_format.append(format_ok)
            except:
                continue

        # 输出结果
        mIoU = sum(total_iou)/len(total_iou)
        mFormat = sum(total_format)/len(total_format)
        res = {"mIoU": mIoU, "format_acc": mFormat}
        logger.info(f"Result: {res}")
        with open(os.path.join(self.config["save_dir"], "coco_grpo_result.json"), "w") as f:
            json.dump(res, f, indent=4)

# ---------------------- 运行 ----------------------
if __name__ == "__main__":
    evaluator = CocoGRPOEvaluator(CONFIG)
    evaluator.run()