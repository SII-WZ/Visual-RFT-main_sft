from coco_evaluation import CocoDetectionEvaluator

evaluator = CocoDetectionEvaluator('/inspire/hdd/global_user/zhouwei-240108540164/autodrive/Visual-RFT-main/share_data/coco_val/annotations/instances_val2017.json')
#results, per_class_results = evaluator.evaluate('./coco_evaluation/2b_sft_prediction.json', './coco_evaluation/results/')
results, per_class_results = evaluator.evaluate('././coco_evaluation/2b_sft_prediction_results_80_wothink_new.json', './coco_evaluation/results/')
### mAP and AP for all categories
results, per_class_results
### mAP and AP for selected categories
selected_cate = ['bus', 'train', 'fire hydrant', 'stop sign', 'cat', 'dog', 'bed', 'toilet']
selected_cate = ['mouse', 'fork', 'hot dog', 'cat', 'airplane', 'suitcase', 'parking meter', 'sandwich', 'train', 'hair drier', 'toilet', 'toaster', 'snowboard', 'frisbee', 'bear']
#selected_cate =['bus','person', 'bicycle', 'car', 'motorcycle',  'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'bench', 'bird', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'skis', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'knife', 'spoon', 'bowl', 'banana', 'apple',  'orange', 'broccoli', 'carrot', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'tv', 'laptop', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',  'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',  'toothbrush']


results, per_class_results
AP_sum = 0
for item in per_class_results:
    for key, value in item.items():
        if key in selected_cate:
            print(f"Key: {key}, Value: {value}")
            AP_sum += value
print("mAP for selected categories: ", (AP_sum)/(len(selected_cate)))