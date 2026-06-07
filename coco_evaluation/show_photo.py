import json
import cv2
import matplotlib.pyplot as plt
import os
from pycocotools.coco import COCO
from sklearn.metrics import average_precision_score
import numpy as np

# 加载 COCO 数据集注释
coco_annotation_path = '/inspire/hdd/global_user/zhouwei-240108540164/autodrive/Visual-RFT-main/share_data/coco_val/annotations/instances_val2017.json'
coco = COCO(coco_annotation_path)

# 加载预测结果
prediction_path = './coco_evaluation/7b_rl_prediction_results.json'  # 使用正确的文件路径
with open(prediction_path, 'r') as f:
    predictions = json.load(f)

# 设置你要分析的图像 ID 和所选类别
image_id = 331352  # 你可以根据需要更改这张图的 ID
selected_cate = ['mouse', 'fork', 'hot dog', 'cat', 'airplane', 'suitcase', 'parking meter', 'sandwich', 'train', 'hair drier', 'toilet', 'toaster', 'snowboard', 'frisbee', 'bear']

# 获取所有类别 ID 和名称
categories = coco.loadCats(coco.getCatIds())
category_names = {category['id']: category['name'] for category in categories}

# 计算每个类别的 AP (平均精度)
def get_ap_for_selected_categories(selected_cate, predictions, category_names, coco, image_id):
    ap_dict = {}
    # 获取该图像的 ground truth 目标框
    ann_ids = coco.getAnnIds(imgIds=image_id)
    anns = coco.loadAnns(ann_ids)
    
    # 对每个类别计算 AP
    for category in selected_cate:
        category_id = None
        # 查找类别 ID
        for cat_id, cat_name in category_names.items():
            if cat_name == category:
                category_id = cat_id
                break
        
        # 获取该类别的预测结果
        category_predictions = [p for p in predictions if p['category_id'] == category_id and p['image_id'] == image_id]
        
        # 获取 ground truth 数据
        category_anns = [ann for ann in anns if ann['category_id'] == category_id]
        
        # 计算该类别的 AP 值
        if category_predictions and category_anns:
            true_boxes = []
            pred_boxes = []
            pred_scores = []
            
            for prediction in category_predictions:
                bbox = prediction['bbox']  # [x, y, width, height]
                score = prediction['score']
                pred_boxes.append(bbox)
                pred_scores.append(score)

            # 对每个ground truth目标
            for ann in category_anns:
                gt_bbox = ann['bbox']
                true_boxes.append(gt_bbox)


            if true_boxes and pred_boxes:
                # 计算AP: 比较 predicted boxes 和 true boxes
                # 比较时应先对预测得分进行排序
                pred_scores = np.array(pred_scores)
                pred_boxes = np.array(pred_boxes)
                true_boxes = np.array(true_boxes)
                
                # 计算每个预测框与真实框的IoU
                def compute_iou(pred_box, gt_box):
                    x1 = max(pred_box[0], gt_box[0])
                    y1 = max(pred_box[1], gt_box[1])
                    x2 = min(pred_box[0] + pred_box[2], gt_box[0] + gt_box[2])
                    y2 = min(pred_box[1] + pred_box[3], gt_box[1] + gt_box[3])

                    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
                    pred_area = pred_box[2] * pred_box[3]
                    gt_area = gt_box[2] * gt_box[3]  # Corrected line

                    union_area = pred_area + gt_area - inter_area

                    # IoU is the intersection area divided by the union area
                    iou = inter_area / union_area
                    return iou
                
                true_boxes_binary = []
                for pred_box in pred_boxes:
                    ious = [compute_iou(pred_box, gt_box) for gt_box in true_boxes]
                    print(ious,"这里")
                    max_iou = max(ious)  # Get the maximum IoU for the current prediction box
                    true_boxes_binary.append(1 if max_iou >= 0.5 else 0)  # Binary value based on IoU threshold

                true_boxes_binary = np.array(true_boxes_binary)
                pred_scores = np.array(pred_scores)

                # 计算 Precision 和 Recall
                ap = average_precision_score(true_boxes_binary.flatten(), pred_scores.flatten())
                ap_dict[category] = ap
            else:
                ap_dict[category] = 0  # 如果没有找到该 p['image_id'] == image = coco.loadImgs(image_id)[0]['file_name']
image_info = coco.loadImgs(image_id)[0]
image_path = f"./share_data/coco_val/val2017/{image_info['file_name']}"
image = cv2.imread(image_path)

# 获取该图像的预测框
image_predictions = [p for p in predictions if p['image_id'] == image_id]

# 在图片上绘制框框和 AP 值
for prediction in image_predictions:
    category_id = prediction['category_id']
    bbox = prediction['bbox']  # [x, y, width, height]
    score = prediction['score']
    
    # 获取类别名称
    category_name = category_names.get(category_id)
    
    if category_name is None:
        print(f"Warning: category_id {category_id} not found in category_names.")
        continue  # 如果找不到该类别，则跳过

    # 只绘制所选类别的框
    if category_name in selected_cate:
        x, y, w, h = [int(i) for i in bbox]
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # 获取每个类别的 AP 值
        ap_dict = get_ap_for_selected_categories(selected_cate, predictions, category_names, coco, image_id)
        if ap_dict is not None:
            ap_value = ap_dict.get(category_name, 0)
        else:
            ap_value = 0  # Or handle appropriately
        
        # 在框上标记 AP 值
        label = f'{category_name}: {ap_value:.2f}'
        
        # 在框的上方显示标签
        cv2.putText(image, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# 保存带有标记的图像
output_path = f'./coco_evaluation/save_image/test_rl_image_{image_id}.jpg'
cv2.imwrite(output_path, image)

print(f"Annotated image saved at {output_path}")
