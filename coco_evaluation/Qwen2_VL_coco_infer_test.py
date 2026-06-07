import pandas as pd
import re
import os

# ===================== 1. 定义关键参数 =====================
# 你的目标检查类别列表
selected_cate = [
    'mouse', 'fork', 'hot dog', 'cat', 'airplane', 
    'suitcase', 'parking meter', 'sandwich', 'train', 
    'hair drier', 'toilet', 'toaster', 'snowboard', 
    'frisbee', 'bear'
]
# Parquet文件路径前缀（分块文件）
file_prefix = "share_data/ViRFT_COCO_base65/data/train-0000"
file_suffix = "-of-00007.parquet"

# ===================== 2. 定义类别提取函数 =====================
def extract_category(problem_text):
    """从problem文本中提取类别名"""
    pattern = r"category '([^']+)'"
    match = re.search(pattern, problem_text)
    if match:
        # 转小写+去空格，避免大小写/空格差异导致匹配失败
        return match.group(1).strip().lower()
    return None

# ===================== 3. 读取所有分块文件，提取所有类别 =====================
all_unique_categories = set()
# 循环读取7个分块文件
for i in range(7):
    file_path = f"{file_prefix}{i}{file_suffix}"
    if not os.path.exists(file_path):
        print(f"⚠️ 文件{file_path}不存在，跳过")
        continue
    
    # 仅读取problem列（节省内存）
    df = pd.read_parquet(file_path, columns=["problem"])
    # 提取类别并转小写
    df["category"] = df["problem"].apply(extract_category)
    # 加入汇总集合（自动去重）
    all_unique_categories.update(df["category"].dropna().unique())

# 转成小写的集合（方便不区分大小写匹配）
all_cate_lower = {cate.lower() for cate in all_unique_categories}

# ===================== 4. 校验selected_cate中的每个类别 =====================
# 记录存在/缺失的类别
exist_cates = []
missing_cates = []

for cate in selected_cate:
    # 转小写匹配，避免大小写问题（比如'Hair Drier' vs 'hair drier'）
    cate_lower = cate.lower()
    if cate_lower in all_cate_lower:
        exist_cates.append(cate)
    else:
        missing_cates.append(cate)

# ===================== 5. 输出校验结果 =====================
print("="*50)
print(f"📊 类别校验结果汇总")
print("="*50)
print(f"指定检查的类别总数：{len(selected_cate)}")
print(f"✅ 存在的类别数量：{len(exist_cates)}")
print(f"❌ 缺失的类别数量：{len(missing_cates)}")

print("\n✅ 存在的类别列表：")
for idx, cate in enumerate(exist_cates, 1):
    print(f"  {idx}. {cate}")

if missing_cates:
    print("\n❌ 缺失的类别列表：")
    for idx, cate in enumerate(missing_cates, 1):
        print(f"  {idx}. {cate}")
else:
    print("\n🎉 所有指定类别都存在于训练数据中！")

# 可选：输出训练数据中所有类别（方便核对）
print("\n" + "="*50)
print(f"📋 训练数据中所有类别（共{len(all_cate_lower)}个）：")
print(sorted(list(all_cate_lower)))