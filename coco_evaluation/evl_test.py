import pandas as pd

# ----------------------
# 你要查看的文件路径
# ----------------------
file_path = "./share_data/ViRFT_COCO_base65/data/train-00000-of-00007.parquet"

# 1. 读取文件
df = pd.read_parquet(file_path)

# 2. 查看数据有多少行多少列
print("="*50)
print("数据形状 (行数, 列数):", df.shape)
print("="*50)

# 3. 查看每一列的名字和类型
print("\n每一列的类型:")
print(df.dtypes)

# 4. 查看第一条完整数据
print("\n===== 第一条数据的完整内容 =====\n")
first_row = df.iloc[0]  # 取第1行
print(first_row["problem"])
print(first_row)

# 5. 专门查看 image 字段是什么格式
print("\n===== 查看 image 字段的格式 =====\n")
image_data = first_row["image"]
print("image 的类型:", type(image_data))  # 看是字符串？字典？还是bytes？
# print("\nimage 内容预览:")
# print(image_data)

# 6. 如果是字典，打印所有key
if isinstance(image_data, dict):
    print("\nimage 是字典，key 有:", list(image_data.keys()))