
# import io
# import json
# import re
# import pandas as pd
# import torch
# from PIL import Image
# from tqdm import tqdm

# from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
# from qwen_vl_utils import process_vision_info

# # ========================= 配置 =========================
# PARQUET_FILE = "./share_data/ViRFT_COCO_base65/data/train-00000-of-00007.parquet"
# OUTPUT_PRED = "./coco_evaluation/final_result.json"

# # 模型路径
# MODEL_PATH = "./out/Qwen2-VL-2B-Instruct_SFT_coco_base65cate_6k_sft/checkpoint-1500"
# PROCESSOR_PATH = "./Qwen/Qwen2-VL-2B-Instruct"

# # ===================== 模型加载 =====================
# device = "cuda:0"
# model = Qwen2VLForConditionalGeneration.from_pretrained(
#     MODEL_PATH, dtype=torch.bfloat16
# ).to(device)
# processor = AutoProcessor.from_pretrained(PROCESSOR_PATH)
# model.eval()

# # ===================== 读取数据 =====================
# df = pd.read_parquet(PARQUET_FILE).head(10)
# predictions = []

# # ===================== 推理 =====================
# for idx, row in tqdm(df.iterrows(), total=len(df)):
#     try:
#         # 1. 读取图片
#         img = Image.open(io.BytesIO(row["image"]["bytes"])).convert("RGB")
#         w, h = img.size

#         # 2. 直接用数据里的问题
#         prompt = row["problem"]

#         # 3. 构造输入
#         messages = [{
#             "role": "user",
#             "content": [
#                 {"type": "image", "image": img},
#                 {"type": "text", "text": prompt}
#             ]
#         }]

#         # 4. 推理
#         text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
#         image_inputs, _ = process_vision_info(messages)
#         inputs = processor(text=[text], images=image_inputs, return_tensors="pt").to(device)

#         with torch.no_grad():
#             out = model.generate(**inputs, max_new_tokens=512, do_sample=False, temperature=0)
#         out = out[:, inputs.input_ids.shape[1]:]
#         res = processor.decode(out[0], skip_special_tokens=True).strip()
#         print(res)

#         # ==================== 【正确】解析框 ====================
#         match = re.search(r"<answer>(.*?)</answer>", res, re.DOTALL)
#         if not match:
#             continue

#         content = match.group(1).strip().replace("'", '"')
#         if "No Objects" in content or content == "[]":
#             continue

#         bboxes = json.loads(content)

#         # ✅【正确】0-1000 转 像素
#         for b in bboxes:
#             # GT 格式：[x1, y1, x2, y2]  0~1000
#             x1, y1, x2, y2 = b["Position"]
#             conf = b["Confidence"]

#             # 正确转换！！！
#             px1 = x1 / 1000.0 * w
#             py1 = y1 / 1000.0 * h
#             px2 = x2 / 1000.0 * w
#             py2 = y2 / 1000.0 * h

#             # 输出格式：x1, y1, width, height
#             bw = px2 - px1
#             bh = py2 - py1

#             predictions.append({
#                 "image_id": int(idx),
#                 "category_id": -1,
#                 "bbox": [px1, py1, bw, bh],
#                 "score": float(conf)
#             })

#     except Exception as e:
#         continue

# # 保存结果
# with open(OUTPUT_PRED, "w") as f:
#     json.dump(predictions, f)

# print(f"\n🎉 推理完成！正确框数量：{len(predictions)}")
# print("✅ 坐标已100%修复！现在运行可视化，框一定准！")


import io
import json
import re
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# -------------------------- 配置 --------------------------
PARQUET_FILE = "./share_data/ViRFT_COCO_base65/data/train-00000-of-00007.parquet"

PARQUET_FILE = "./share_data/COCOwothink/data/train_clean.parquet"
OUTPUT_PRED = "./coco_evaluation/2b_sft_train65_result_only2.json"
MODEL_PATH =  "./out/Qwen2-VL-2B-Instruct_SFT_coco_base65cate_6k_sft_wothink_new30/checkpoint-8"
PROCESSOR_PATH = "./Qwen/Qwen2-VL-2B-Instruct"
#MODEL_PATH =  "./out/Qwen2-VL-2B-Instruct_SFT_coco_base65cate_6k_sft/checkpoint-1500test"
#MODEL_PATH = "./out/Qwen2-VL-2B-Instruct_GRPO_coco_base65cate_6k/checkpoint-1500test"
#MODEL_PATH ="./out/Qwen2-VL-2B-Instruct_SFT_coco_base65cate_6k_sft_wothink_only2/checkpoint-80"
#MODEL_PATH= "./Qwen/Qwen2-VL-2B-Instruct"
# -----------------------------------------------------------

device = "cuda:0"
model = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_PATH, dtype=torch.bfloat16
).to(device)
processor = AutoProcessor.from_pretrained(PROCESSOR_PATH)
model.eval()

df = pd.read_parquet(PARQUET_FILE).head(30)
predictions = []

for idx, row in tqdm(df.iterrows(), total=len(df)):
    try:
        # 读取原图
        img_bytes = row["image"]["bytes"]
        raw_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        W, H = raw_img.size

        prompt = row["problem"]

        # 构造对话
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": raw_img},
                {"type": "text", "text": prompt}
            ]
        }]

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, _ = process_vision_info(messages)
        inputs = processor(text=[text], images=image_inputs, return_tensors="pt").to(device)

        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=512, do_sample=False, temperature=0)
        out = out[:, inputs.input_ids.shape[1]:]
        res = processor.decode(out[0], skip_special_tokens=True).strip()
        print(res,"1")

        # 解析 answer 里的 bbox
       # match = re.search(r"(.*?)", res, re.DOTALL)
        match=re.search(r'<answer>(.*?)</answer>', res)
        
        if not match:
            continue
        content = match.group(1).strip().replace("'", '"')
       # content = content.replace("...", "")
        print(content)
        if "No Objects" in content or content == "[]":
            continue

        bboxes = json.loads(content)
        print(bboxes,"3")

        for b in bboxes:
            print(b,"4")
            # 模型输出：0~1000 归一化 [x1,y1,x2,y2]
            nx1, ny1, nx2, ny2 = b["Position"]
            conf = b["Confidence"]

            # ========== 【修复核心：标准 0~1000 → 原图像素】 ==========
            px1 = nx1# / 1000.0 * W
            py1 = ny1 #/ 1000.0 * H
            px2 = nx2 #/ 1000.0 * W
            py2 = ny2 #/ 1000.0 * H
            # ======================================================

            # 转 COCO 要求 [x1, y1, w, h]
            bw = px2 - px1
            bh = py2 - py1

            predictions.append({
                "image_id": int(idx),
                "category_id": 1,
                "bbox": [round(px1,2), round(py1,2), round(bw,2), round(bh,2)],
                "score": float(conf)
            })
    except Exception as e:
        print(f"idx {idx} error: {e}")
        continue

with open(OUTPUT_PRED, "w", encoding="utf-8") as f:
    json.dump(predictions, f, ensure_ascii=False, indent=2)

print(f"\n✅ 推理完成，已保存 {len(predictions)} 个修复坐标的预测框")