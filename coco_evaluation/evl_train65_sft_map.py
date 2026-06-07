# import json
# import pandas as pd
# import numpy as np

# # # ==================== 路径 ====================
# PREDICT_JSON = "./coco_evaluation/final_result.json"
# PARQUET_FILE = "./share_data/ViRFT_COCO_base65/data/train-00000-of-00007.parquet"
# PARQUET_FILE = "./share_data/COCOwothink/data/train_clean.parquet"

# # 你的65类
# selected_cate = [
#     'person', 'bicycle', 'car', 'motorcycle', 'truck', 'boat',
#     'traffic light', 'fire hydrant', 'stop sign', 'bench',
#     'bird', 'dog', 'horse', 'sheep', 'cow', 'elephant',
#     'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag',
#     'tie', 'skis', 'sports ball', 'kite', 'baseball bat',
#     'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
#     'bottle', 'wine glass', 'cup', 'knife', 'spoon', 'bowl',
#     'banana', 'apple', 'orange', 'broccoli', 'carrot',
#     'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant',
#     'bed', 'dining table', 'tv', 'laptop', 'remote', 'keyboard',
#     'cell phone', 'microwave', 'oven', 'sink', 'refrigerator',
#     'book', 'clock', 'vase', 'scissors', 'teddy bear',
#     'hair drier', 'toothbrush'
# ]

# # ==================== 加载预测结果 ====================
# with open(PREDICT_JSON, "r") as f:
#     predictions = json.load(f)

# # ==================== 加载真实标签（parquet） ====================
# df = pd.read_parquet(PARQUET_FILE)

# # ==================== 从 problem 里提取类别 ====================
# def extract_category_from_problem(problem):
#     for c in selected_cate:
#         if f"'{c}'" in problem:
#             return c
#     return None

# # ==================== 计算 IoU ====================
# def iou(box1, box2):
#     x1, y1, x2, y2 = box1
#     x1_, y1_, x2_, y2_ = box2

#     xi1 = max(x1, x1_)
#     yi1 = max(y1, y1_)
#     xi2 = min(x2, x2_)
#     yi2 = min(y2, y2_)

#     if xi2 <= xi1 or yi2 <= yi1:
#         return 0.0

#     inter = (xi2 - xi1) * (yi2 - yi1)
#     area1 = (x2 - x1) * (y2 - y1)
#     area2 = (x2_ - x1_) * (y2_ - y1_)
#     union = area1 + area2 - inter
#     return inter / union

# # ==================== 统计正确框 ====================
# total_gt = 0
# total_det = 0
# true_pos = 0

# for idx, row in df.iterrows():
#     # 真实框
#     try:
#         gt_boxes = json.loads(row["solution"].replace("'", '"').replace("<answer>", "").replace("</answer>", ""))
#     except:
#         gt_boxes = []

#     # 预测框（只取这张图）
#     pred_boxes = [p for p in predictions if p["image_id"] == idx]

#     # 计算匹配
#     for gt in gt_boxes:
#         total_gt += 1
#         gt_pos = gt["Position"]
#         matched = False
#         for p in pred_boxes:
#             pb = p["bbox"]
#             pred_pos = [pb[0], pb[1], pb[0]+pb[2], pb[1]+pb[3]]
#             if iou(gt_pos, pred_pos) > 0.5:
#                 true_pos += 1
#                 matched = True
#                 break
#     total_det += len(pred_boxes)

# # ==================== 计算 mAP ====================
# precision = true_pos / total_det if total_det > 0 else 0
# recall = true_pos / total_gt if total_gt > 0 else 0
# f1 = 2 * precision * recall / (precision + recall + 1e-8)

# print("======= 训练集评估结果 (正确版) =======")
# print(f"真实框数量: {total_gt}")
# print(f"预测框数量: {total_det}")
# print(f"正确匹配: {true_pos}")
# print(f"Precision: {precision:.4f}")
# print(f"Recall: {recall:.4f}")
# print(f"F1(mAP近似): {f1:.4f}")


###############这边是查看内部信息的
import pandas as pd

# 你的文件路径
# PARQUET_PATH = "./share_data/ViRFT_COCO_base65/data/train_clean.parquet"

# # 读取数据
# df = pd.read_parquet(PARQUET_PATH)

# # 打印前 10 条的 问题(problem) 和 答案(solution)
# print("=" * 80)
# # print("📝 查看 parquet 前 10 条数据：问题 + 答案")
# print("=" * 80)

# # 取前 10 条
# for i in range(1):
#     row = df.iloc[i]
#     print(row)
    
    
#     print(f"\n🟢 第 {i} 条数据")
#     print("-" * 60)
    
#     # 打印完整问题
#     print("🔍 询问语句 (problem)：")
#     print(row["problem"])
    
#     print("\n✅ 回答语句 (solution)：")
#     print(row["solution"])
    
#     print("=" * 80)
    
    
# ################## 这边是画出框
# import io
# import json
# import os
# import pandas as pd
# from PIL import Image, ImageDraw

# PARQUET_PATH = "./share_data/ViRFT_COCO_base65/data/train-00000-of-00007.parquet"
# PRED_JSON_PATH = "./coco_evaluation/final_result.json"
# PARQUET_FILE = "./share_data/COCOwothink/data/train_clean.parquet"
# SAVE_FOLDER = "./coco_evaluation/save_image"
# START_IDX = 0
# END_IDX = 10

# os.makedirs(SAVE_FOLDER, exist_ok=True)
# df = pd.read_parquet(PARQUET_PATH)
# with open(PRED_JSON_PATH, "r", encoding="utf-8") as f:
#     preds = json.load(f)

# def get_gt(sol):
#     try:
#         s = sol.replace("","").replace("","").replace("'",'"')
#         return json.loads(s)
#     except:
#         return []

# for idx in range(START_IDX, END_IDX):
#     try:
#         row = df.iloc[idx]
#         img_bytes = row["image"]["bytes"]
#         img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
#         W, H = img.size
#         draw = ImageDraw.Draw(img)

#         # 真实GT
#         gt_boxes = get_gt(row["solution"])
#         print(gt_boxes)
#         # 当前图预测
#         cur_preds = [p for p in preds if p["image_id"] == idx]
#         print(cur_preds)
#         # 画红色GT
#         for g in gt_boxes:
#             nx1, ny1, nx2, ny2 = g["Position"]
#             x1 = nx1 / 1000.0 * W
#             y1 = ny1 / 1000.0 * H
#             x2 = nx2 / 1000.0 * W
#             y2 = ny2 / 1000.0 * H
#             print(x1,x2,y1,y2)
#             draw.rectangle([x1, y1, x2, y2], outline="red", width=3)

#         # 画绿色预测
#         for p in cur_preds:
#             x1, y1, bw, bh = p["bbox"]
#             x2 = x1 + bw
#             y2 = y1 + bh
#             print(x1,x2,y1,y2)
#             draw.rectangle([x1, y1, x2, y2], outline="green", width=3)

#         save_path = os.path.join(SAVE_FOLDER, f"vis_fix_{idx}.jpg")
#         img.save(save_path)
#         print(f"✅ 已保存第 {idx} 张: {save_path}")
#     except Exception as e:
#         print(f"❌ 第 {idx} 张失败: {e}")

# print("\n🎉 全部可视化完成，坐标已修复对齐！")



####################修改que文件
# import pandas as pd
# import re

# # 1. 你的原始 parquet 路径
# INPUT_PARQUET = "./share_data/ViRFT_COCO_base65/data/train-00006-of-00007.parquet"

# # 2. 输出新的干净文件（不会覆盖原文件，安全！）
# OUTPUT_PARQUET = "./share_data/ViRFT_COCO_base65/data/train_clean6.parquet"

# # --------------------------------------------------------------------
# # 你想要的【新版无 think prompt】
# # --------------------------------------------------------------------
# NEW_PROMPT_PREFIX = (
#     "Detect all objects belonging to the category '{}' in the image, and provide the bounding boxes (between 0 and 1000, integer) and confidence (between 0 and 1, with two two decimal places).\n"
#     "If no object belonging to the category '{}' in the image, return 'No Objects'.\n"
#     "Output the final answer in <answer> </answer> tags. The output answer format should be as follows:\n"
#     "<answer>[{{'Position': [x1, y1, x2, y2], 'Confidence': number}}, ...]</answer>\n"
#     "Please strictly follow the format."
# )

# # 读取数据集
# df = pd.read_parquet(INPUT_PARQUET)

# # 批量替换所有 problem 字段
# def clean_problem_text(old_problem):
#     try:
#         # 提取 category 名称（比如 person）
#         match = re.search(r"category '(.+?)'", old_problem)
#         if not match:
#             return old_problem
#         cate = match.group(1)
        
#         # 返回新的、干净的、无 think 的 prompt
#         return NEW_PROMPT_PREFIX.format(cate, cate)
#     except:
#         return old_problem

# # 执行替换
# df["problem"] = df["problem"].apply(clean_problem_text)

# # 保存新的 parquet
# df.to_parquet(OUTPUT_PARQUET, index=False)

# print("✅ 数据集修改完成！已生成 clean 版本：", OUTPUT_PARQUET)
# print("✅ 所有 prompt 已去掉 think，只保留 <answer> 输出！")





   
# # ################## 这边是去掉think画出框
# import io
# import json
# import os
# import pandas as pd
# from PIL import Image, ImageDraw

# PRED_JSON_PATH = "./coco_evaluation/final_result.json"
# PARQUET_PATH = "./share_data/COCOwothink/data/train_clean.parquet"
# SAVE_FOLDER = "./coco_evaluation/save_image"
# START_IDX = 0
# END_IDX = 40

# os.makedirs(SAVE_FOLDER, exist_ok=True)
# df = pd.read_parquet(PARQUET_PATH)
# with open(PRED_JSON_PATH, "r", encoding="utf-8") as f:
#     preds = json.load(f)


# def get_gt(sol):
#     try:
#         # 👇 关键：把 <answer> 和 </answer> 删掉！
#         s = sol.replace("<answer>", "").replace("</answer>", "").replace("'", '"')
#         return json.loads(s)
#     except:
#         return []

# for idx in range(START_IDX, END_IDX):
#     try:
#         row = df.iloc[idx]
#         img_bytes = row["image"]["bytes"]
#         img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
#         W, H = img.size
#         draw = ImageDraw.Draw(img)

#         # 真实GT
#         gt_boxes = get_gt(row["solution"])
#         #print(gt_boxes,"solution")
#         #print(row)
#         #print(row["problem"])
#         # 当前图预测
#         cur_preds = [p for p in preds if p["image_id"] == idx]
#         #print(cur_preds)
#         # 画红色GT
#         for g in gt_boxes:
#             nx1, ny1, nx2, ny2 = g["Position"]
#             x1 = nx1 / 1000.0 * W
#             y1 = ny1 / 1000.0 * H
#             x2 = nx2 / 1000.0 * W
#             y2 = ny2 / 1000.0 * H
#             print([x1,x2,y1,y2])
#             draw.rectangle([x1, y1, x2, y2], outline="red", width=3)

#     #    画绿色预测
#         for p in cur_preds:
#             x1, y1, bw, bh = p["bbox"]
#             print(x1, y1, bw, bh)
#             x2 = x1 + bw
#             y2 = y1 + bh

#             x1 = x1 / 1000.0 * W
#             y1 = y1 / 1000.0 * H
#             x2 = x2 / 1000.0 * W
#             y2 = y2 / 1000.0 * H
#             print([x1,x2,y1,y2])
#             draw.rectangle([x1, y1, x2, y2], outline="green", width=3)

#         save_path = os.path.join(SAVE_FOLDER, f"vis_fix_{idx}.jpg")
#         img.save(save_path)
#         print(f"✅ 已保存第 {idx} 张: {save_path}")
#     except Exception as e:
#         print(f"❌ 第 {idx} 张失败: {e}")

# print("\n🎉 全部可视化完成，坐标已修复对齐！")

########这是去掉think计算map的
import json
import pandas as pd
import numpy as np

# ==================== 路径 ====================
PREDICT_JSON = "./coco_evaluation/2b_sft_train65_result_only2.json"
PARQUET_FILE = "./share_data/COCOwothink/data/train_clean.parquet"
#PARQUET_FILE = "./share_data/ViRFT_COCO_base65/data/train-00000-of-00007.parquet"

# 65类
selected_cate = [
    'person', 'bicycle', 'car', 'motorcycle', 'truck', 'boat',
    'traffic light', 'fire hydrant', 'stop sign', 'bench',
    'bird', 'dog', 'horse', 'sheep', 'cow', 'elephant',
    'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag',
    'tie', 'skis', 'sports ball', 'kite', 'baseball bat',
    'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'orange', 'broccoli', 'carrot',
    'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant',
    'bed', 'dining table', 'tv', 'laptop', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'sink', 'refrigerator',
    'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'bus', 'toothbrush'
]

# ==================== 工具函数 ====================
def get_gt(sol):
    try:
        s = sol.replace("<answer>", "").replace("</answer>", "").replace("'", '"')
        return json.loads(s)
    except:
        return []

def iou(box1, box2):
    x1, y1, x2, y2 = box1
    x1_, y1_, x2_, y2_ = box2
    xi1 = max(x1, x1_)
    yi1 = max(y1, y1_)
    xi2 = min(x2, x2_)
    yi2 = min(y2, y2_)
    if xi2 <= xi1 or yi2 <= yi1:
        return 0.0
    inter = (xi2 - xi1) * (yi2 - yi1)
    area1 = (x2 - x1) * (y2 - y1)
    area2 = (x2_ - x1_) * (y2_ - y1_)
    union = area1 + area2 - inter
    return inter / union

# ==================== 加载数据 ====================
with open(PREDICT_JSON, "r") as f:
    predictions = json.load(f)
df = pd.read_parquet(PARQUET_FILE).head(30)

# ==================== 关键参数 ====================

IOU_THRESH = 0.5

# ==================== 评估开始 ====================
total_gt = 0
total_det = 0
true_pos = 0

for idx, row in df.iterrows():
    print(row["problem"])
    gt_boxes = get_gt(row["solution"])
    pred_boxes = [p for p in predictions if p["image_id"] == idx]

    # 记录已匹配的GT，防止重复匹配
    matched_gts = [False] * len(gt_boxes)

    # 遍历预测框
    for p in pred_boxes:
        total_det += 1
        pb = p["bbox"]
        x1, y1, w, h = pb
        x2 = x1 + w
        y2 = y1 + h

        # 🔥 缩放：640 → 1000
        x1_s = x1 
        y1_s = y1 
        x2_s = x2 
        y2_s = y2 
        pred_pos = [x1_s, y1_s, x2_s, y2_s]

        # 找匹配的GT
        max_iou = 0
        max_idx = -1
        for i, gt in enumerate(gt_boxes):
            if matched_gts[i]:
                continue
            gt_pos = gt["Position"]
            iou_val = iou(pred_pos, gt_pos)
            if iou_val > max_iou:
                max_iou = iou_val
                max_idx = i

        if max_iou >= IOU_THRESH and max_idx >= 0:
            true_pos += 1
            matched_gts[max_idx] = True  # 标记已匹配

    total_gt += len(gt_boxes)

# ==================== 计算指标 ====================
precision = true_pos / total_det if total_det > 0 else 0
recall = true_pos / total_gt if total_gt > 0 else 0
f1 = 2 * precision * recall / (precision + recall + 1e-8)

print("======= 正确评估结果（坐标已对齐）=======")
print(f"真实框: {total_gt}")
print(f"预测框: {total_det}")
print(f"正确匹配: {true_pos}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")