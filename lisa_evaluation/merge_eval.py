import json
import os
merged = []
# for i in range(int(os.environ['SPLIT_NUM'])):
#     data = json.load(open(f"tmp/res_{i}.json", 'r'))
#     merged += data
# print(f"mIoU: {sum(merged) / len(merged)}")

merged = []

data = json.load(open(f"/inspire/hdd/global_user/zhouwei-240108540164/autodrive/Visual-RFT-main/lisa_evaluation/2b_ori_test/res_0.json", 'r'))
valid_data =data # [num for num in data if num != 0] 
print(f"mIoU: {sum(valid_data )/(len(valid_data)-2)}")

