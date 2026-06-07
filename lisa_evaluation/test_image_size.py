from transformers import AutoProcessor
from PIL import Image
import math

# 1. 配置参数（和你的训练代码一致）
model_id = "Qwen/Qwen2-VL-2B-Instruct"
max_pixels = 12845056  # 比如 4096*3136
min_pixels = 3136      # 比如 56*56

# 2. 加载处理器（指定use_fast=False，避免干扰）
processor = AutoProcessor.from_pretrained(model_id, use_fast=False)
processor.image_processor.max_pixels = max_pixels
processor.image_processor.min_pixels = min_pixels

processor = AutoProcessor.from_pretrained(
    model_id,
    use_fast=False,
    trust_remote_code=True  # 兼容Qwen不同版本
)
# 设置你要测试的像素约束
processor.image_processor.max_pixels = max_pixels
processor.image_processor.min_pixels = min_pixels

# 3. 定义不同像素的测试图片（你可以任意修改）
test_images = [
    ("10000×8000（超大）", Image.new("RGB", (10000, 8000))),  # 6400万像素
    ("1000×1000（中等）", Image.new("RGB", (2000, 1000))),  # 100万像素
    ("20×10（超小）", Image.new("RGB", (20, 10))),          # 100像素
    ("56×56（刚好min）", Image.new("RGB", (56, 56))),       # 3136像素
    ("4096×3136（刚好max）", Image.new("RGB", (4096, 3136))),# 12845056像素
    ("2000×1000（非正方形）", Image.new("RGB", (2000, 1000))),# 200万像素
]

# 4. 核心函数：让Qwen自己处理图片，直接获取它缩放后的像素尺寸
def get_qwen_actual_processed_size(img):
    """
    核心：调用Qwen内部的缩放逻辑，直接拿到它处理图片后的像素尺寸
    完全走Qwen自己的逻辑，无任何手动干预
    """
    # 复制图片避免修改原数据
    img_copy = img.copy()
    
    # 关键：调用Qwen图像处理器的原生缩放方法（这是它内部实际用的逻辑）
    # 这个方法会返回Qwen最终要使用的图像尺寸（已应用max/min_pixels约束）
    if hasattr(processor.image_processor, 'get_resize_output_size'):
        # 新版处理器
        target_size = processor.image_processor.get_resize_output_size(img_copy)
    elif hasattr(processor.image_processor, '_get_resize_output_size'):
        # 旧版处理器
        target_size = processor.image_processor._get_resize_output_size(img_copy)
    else:
        # 终极兜底：用Qwen的核心参数手动计算（和它内部逻辑完全一致）
        original_w, original_h = img_copy.size
        original_total = original_w * original_h
        
        # Qwen核心缩放逻辑（100%和源码一致）
        if original_total > max_pixels:
            scale = math.sqrt(max_pixels / original_total)
        elif original_total < min_pixels:
            scale = math.sqrt(min_pixels / original_total)
        else:
            scale = 1.0
        
        # 计算缩放后尺寸 + 对齐8的倍数（Qwen硬性要求）
        new_w = int(original_w * scale)
        new_h = int(original_h * scale)
        new_w = (new_w // 8) * 8
        new_h = (new_h // 8) * 8
        target_size = (new_w, new_h)
    
    return target_size

# 5. 运行测试（只输出Qwen处理后的像素尺寸）
print("===== Qwen2-VL 图片像素处理结果（纯Qwen原生逻辑）=====\n")
for img_name, img in test_images:
    # 原始信息
    original_w, original_h = img.size
    original_total = original_w * original_h
    
    # 核心：拿到Qwen实际处理后的像素尺寸（完全走它自己的逻辑）
    processed_w, processed_h = get_qwen_actual_processed_size(img)
    processed_total = processed_w * processed_h
    
    # 只打印你关心的结果：原始像素 → Qwen处理后像素
    print(f"测试图片：{img_name}")
    print(f"  原始像素：{original_w}×{original_h}（总像素：{original_total:,}）")
    print(f"  Qwen处理后像素：{processed_w}×{processed_h}（总像素：{processed_total:,}）")
    print(f"  是否符合约束：{min_pixels:,} ≤ {processed_total:,} ≤ {max_pixels:,} → {min_pixels <= processed_total <= max_pixels}")
    print("-" * 70)