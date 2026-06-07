import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import json
import os
from PIL import Image
import logging
from tqdm import tqdm
import re
# from process_utils import pred_2_point, extract_bbox
import math
response=""
pattern = r"\(\s*(\d+)\s*,\s*(\d+)\s*\)\s*,\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)"
pattern =  r"\(\s*(\d+)\s*[，,]\s*(\d+)\s*(?:[，,]\s*\(\s*|\s*[，,]\s*)\s*(\d+)\s*[，,]\s*(\d+)\s*\)"   #修改后的正择表达式
matches = re.findall(pattern, response)
x1, y1, x2, y2 = map(int, matches[0])
