from datasets import load_dataset, load_from_disk, Dataset, concatenate_datasets
import PIL
from PIL import Image
def make_conversation_image(example):
        conversations = example["conversations"]
        formatted_conversation = []
        for message in conversations:
            if message["from"] == "user":
                # Check if the message contains an image
                if "<img>" in message["value"] and "</img>" in message["value"]:
                    image_path = message["value"].split("<img>")[1].split("</img>")[0].strip()
                    text_content = message["value"].replace(f"<img>{image_path}</img>", "").strip()
                    formatted_conversation.append(
                        {
                            "role": "user",
                            "content": [
                                {"type": "image"},
                                {"type": "text", "text": QUESTION_TEMPLATE.format(Question=text_content)},
                            ],
                        }
                    )
                else:
                    formatted_conversation.append({"role": "user", "content": message["value"]})
            elif message["from"] == "assistant":
                res = '<answer> ' + message["value"] + ' </answer>'
                # formatted_conversation.append({"role": "assistant", "content": message["value"]})
        # print(image_path)
        return {"image": Image.open(image_path), "prompt": formatted_conversation, 'solution': res}

QUESTION_TEMPLATE = "{Question} Output the thinking process in <think> </think> and your grouding box. Following \"<think> thinking process </think>\n<answer>(x1,y1),(x2,y2)</answer>)\" format."

dataset_val = Dataset.from_json("/inspire/hdd/global_user/zhouwei-240108540164/autodrive/Visual-RFT-main/share_data/lisa_val_sft.json")
    # ... 原有数据处理逻辑（make_conversation_image等） ...
print(type(dataset_val))
dataset_val = dataset_val.map(make_conversation_image)
print(dataset_val)


# col_based_dict = dataset_val[0:4]

# # 2. 提取所有列名和对应数据列表
# columns = list(col_based_dict.keys())
# # 确保所有列的长度一致（数据完整性校验）
# row_count = len(col_based_dict[columns[0]])
# for col in columns:
#     if len(col_based_dict[col]) != row_count:
#         raise ValueError(f"列 {col} 数据长度与其他列不一致，数据损坏")

# # 3. 转换为「行优先」字典列表
# row_based_list = []
# for i in range(row_count):
#     row_dict = {col: col_based_dict[col][i] for col in columns}
#     row_based_list.append(row_dict)

# # 4. 此时即可正常使用原有遍历逻辑，无报错
# prompts = [x["prompt"] for x in row_based_list]
# top4_prompts = prompts[:4]


#print(type(first_data))
#print(first_data)
#print(first_data.dtypes)
# print(first_data)
# prompts = [x["prompt"] for x in first_data]
# print(type(prompts))
# print("第0条数据的格式：", type(first_data))  # 输出：<class 'dict'>
# print("第0条数据的prompt：", first_data["prompt"])
# print("第0条数据的image类型：", type(first_data["image"]))  # 输出：<class 'PIL.Image.Image'>
# print("第0条数据的solution：", first_data["solution"])

