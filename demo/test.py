import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import json
import os
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
import re
import math

torch.manual_seed(1234)
img2description = dict()

SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)


def prepare_inputs(img_path, instruction):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img_path},
                {"type": "text", "text": f"Output the bounding box in the image corresponding to the instruction: {instruction}. Output the thinking process in <think> </think> and output the location of the target's bounding box in <answer> (x1,y1),(x2,y2) </answer> ,input image coordinates where x, y range from 0 to 1000. Following \"<think> thinking process </think>\n<answer>(x1,y1),(x2,y2)</answer>)\" format."}
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


# model = Qwen2VLForConditionalGeneration.from_pretrained(
#     "Zery/Qwen2-VL-7B_visual_rft_lisa_IoU_reward", device_map="auto"
# ).eval()

# processor = AutoProcessor.from_pretrained("Zery/Qwen2-VL-7B_visual_rft_lisa_IoU_reward")
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-7B-Instruct", torch_dtype=torch.bfloat16,device_map="auto",local_files_only=True
).eval()

processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct",local_files_only=True)


image_path = "/inspire/hdd/global_user/zhouwei-240108540164/autodrive/Visual-RFT-main/share_data/lisa/reasonseg/test/3573115220_1394e3cd7c_o.jpg" #"../assets/wash_hands.jpg"
inputs = prepare_inputs(image_path, "Flying in the air can avoid many obstacles and greatly improve commuting efficiency. What form of transportation in the picture can accomplish this?")   #"the pokeymon that can perform Thunderbolt. Output thinking process as detail as possibile"

with torch.no_grad():
    generated_ids = model.generate(**inputs, max_new_tokens=128)
response = processor.batch_decode(
    generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
)[0]
print(response)


pattern = r"\(\s*(\d+)\s*,\s*(\d+)\s*\)\s*,\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)"
matches = re.findall(pattern, response)
image = Image.open(image_path).convert("RGB")
draw = ImageDraw.Draw(image)
w, h = Image.open(image_path).size
x1, y1, x2, y2 = map(int, matches[0])
box_r1 = [int(x1) / 1000, int(y1) / 1000, int(x2) / 1000, int(y2) / 1000]
draw = ImageDraw.Draw(image)
draw.rectangle([box_r1[0] * w, box_r1[1] * h, box_r1[2] * w, box_r1[3] * h], outline="green", width=5)
image
image.save("output_with_box.jpg")
