import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import json
import os
from PIL import Image
import logging
from tqdm import tqdm
import re
# from process_utils import pred_2_point, extract_bbox
import math

logging.basicConfig(level=logging.INFO)
torch.manual_seed(1234)
img2description = dict()

SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)

# Load Qwen2-VL-2B model and processor
# model = Qwen2VLForConditionalGeneration.from_pretrained(
#     "/path/to/your/checkpoint-498", torch_dtype=torch.bfloat16, device_map="auto", attn_implementation="flash_attention_2"
# ).eval()

# processor = AutoProcessor.from_pretrained("/path/to/your//checkpoint-498")

# model = Qwen2VLForConditionalGeneration.from_pretrained(
#     "../Zery/Qwen2-VL-7B_visual_rft_lisa_IoU_reward", device_map="auto",local_files_only=True
# ).eval()

# processor = AutoProcessor.from_pretrained("../Zery/Qwen2-VL-7B_visual_rft_lisa_IoU_reward",local_files_only=True)

model = Qwen2VLForConditionalGeneration.from_pretrained(
    "../Qwen/Qwen2-VL-7B-Instruct", device_map="auto",local_files_only=True
).eval()

processor = AutoProcessor.from_pretrained("../Qwen/Qwen2-VL-7B-Instruct",local_files_only=True)

logging.info("Model and processor loaded successfully")

def process_image(image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    return Image.open(image_path)

def prepare_inputs(img_path, instruction):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img_path},
                {"type": "text", "text": f"Output the bounding box in the image corresponding to the instruction: {instruction}. Output the thinking process in <think> </think> and your grouding box in <answer> </answer>. Following \"<think> thinking process </think>\n<answer>(x1,y1),(x2,y2)</answer>)\" format."}
             #   {"type": "text", "text": f"Output the bounding box in the image corresponding to the instruction: {instruction}. Output the thinking process in <think> </think> and your grouding box.Output the thinking process in \"<think> thinking process </think> and your bounding box in the format of (x1,y1),(x2,y2)\""}
            ]
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        padding=True,
        return_tensors="pt",
    )
    return inputs.to("cuda")

def extract_bbox(response):
    try:
        match = re.search(r"\[(\d+),(\d+),(\d+),(\d+)\]", response)
        if match:
            return [int(match.group(i)) for i in range(1, 5)]
        else:
            raise ValueError("Invalid response format")
    except Exception as e:
        logging.error(f"Error extracting bbox: {e}")
        return None

def compute_iou(boxA, boxB):
    """
    计算 IoU (Intersection over Union)
    boxA, boxB 格式: [x1, y1, x2, y2]
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return iou

def evaluate_model(tasks,save_name="test", split=0, split_num=1):  # 接收参数
    results = []
    box_res =[]
    for task in tasks:
        logging.info(f"Processing task: {task}")
        ious = []
        screenspot_data = json.load(open(f"../share_data/lisa_{task}.json", 'r'))      
        # 替换环境变量读取为函数参数
        data_per_gpu = math.ceil(len(screenspot_data) / split_num)
        start_idx = split * data_per_gpu
        end_idx = min(start_idx + data_per_gpu, len(screenspot_data))
        screenspot_data = screenspot_data[start_idx:end_idx]
        print(f"SPLIT={split} 有 {len(screenspot_data)} 条数据")  # 打印参数

        
        for item in tqdm(screenspot_data):
            img_path = item['image_path']
            try:
                image = process_image(img_path)
                w, h = image.size
                instruction = item["instruction"][0]
                bbox = item["boxes"][0]
                bbox = [
                    bbox[0] / image.size[0],
                    bbox[1] / image.size[1],
                    bbox[2] / image.size[0],
                    bbox[3] / image.size[1],
                ]
                inputs = prepare_inputs(img_path, instruction)

                with torch.no_grad():
                    generated_ids = model.generate(**inputs, max_new_tokens=128)
                response = processor.batch_decode(
                    generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )[0]
                print(response)
                pattern = r"\(\s*(\d+)\s*,\s*(\d+)\s*\)\s*,\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)"
                matches = re.findall(pattern, response)
                x1, y1, x2, y2 = map(int, matches[0])
                pred_bbox = [int(x1) / 1000, int(y1) / 1000, int(x2) / 1000, int(y2) / 1000]

                iou = compute_iou(pred_bbox, bbox)
                box_res.append(
                    {
                        "image_pth": item['image_path'],
                        "pred_bbox": pred_bbox,
                        "thinking_process": response
                    }
                )
                ious.append(iou)
            except Exception as e:
                
                logging.error(f"处理图片 {img_path} 失败：{str(e)}")
                    # 可选：把异常数据也写入box_res（标注为失败）
                box_res.append(
                    {
                    "image_pth": img_path,
                    "pred_bbox": [0.0, 0.0, 0.0, 0.0],  # 标记为失败
                    "thinking_process": f"Error: {str(e)}",
                    "status": "failed"
                    }
                )
                ious.append(0)
        os.makedirs(save_name, exist_ok=True)
        json.dump(box_res, open(f"{save_name}/resbox_{split}_r1_w_think_7b.json", 'w'), indent=4)
        json.dump(ious, open(f"{save_name}/res_{split}.json", 'w'), indent=4)
    return

if __name__ == "__main__":
    # import argparse

    # parser = argparse.ArgumentParser()
    # parser.add_argument('--task', type=str, required=True)
    # args = parser.parse_args()

    # if args.task == "all":
    #     tasks = ["test"]
    # else:
    #     tasks = [args.task]
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default="test_error", help='任务名称')
    parser.add_argument('--split', type=int, default=0, help='当前分片索引（从0开始）')
    parser.add_argument('--split_num', type=int, default=2, help='总分片数')
    parser.add_argument('--save_name', type=str, default="7b_ori_test", help='保存的文件夹名字')
    args = parser.parse_args()

    tasks = [args.task]
    save_name=args.save_name
    # tasks = ["test_error"]
    results = evaluate_model(tasks, save_name=save_name)
