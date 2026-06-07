# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import torch
import os
from dataclasses import dataclass, field
from typing import Optional

from datasets import load_dataset
from transformers import (
    Qwen2VLForConditionalGeneration,
    AutoProcessor,
    TrainingArguments,
    Trainer,
    default_data_collator,
)
from peft import get_peft_config, get_peft_model
from trl import ModelConfig, TrlParser, ScriptArguments

@dataclass
class SFTScriptArguments(ScriptArguments):
    max_pixels: Optional[int] = field(default=12845056)
    min_pixels: Optional[int] = field(default=3136)
    max_seq_length: Optional[int] = field(default=2048)

SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within   and <answer> </answer> tags."
)

def main(script_args, training_args, model_args):
    # 加载模型
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype="auto",
        attn_implementation=model_args.attn_implementation,
        trust_remote_code=True
    )

    processor = AutoProcessor.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=True,
        max_pixels=script_args.max_pixels,
        min_pixels=script_args.min_pixels,
    )
    processor.tokenizer.padding_side = "right"

    # LoRA
    if model_args.use_peft:
        model = get_peft_model(model, get_peft_config(model_args))
        model.print_trainable_parameters()

    # ===================== 完全照搬 GRPO 数据处理 =====================
   # dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config) #运动验证数据集名字
    dataset = load_dataset(script_args.dataset_name, data_dir="data")
    def make_conversation(example):
        return {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": example["problem"]},
            ],
        }

    def make_conversation_image(example):
        return {
            "prompt": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": example["problem"]},
                    ],
                },
            ],
        }

    if "image" in dataset[script_args.dataset_train_split].features:
        print("has image in dataset")
        dataset = dataset.map(make_conversation_image)
    else:
        print("no image in dataset")
        dataset = dataset.map(make_conversation)
        dataset = dataset.remove_columns("messages")


    # def tokenize_fn(examples):
    #     # 批量对话
    #     texts = [processor.apply_chat_template(p, tokenize=False) for p in examples["prompt"]]

    #     # 必须 pt，否则 Qwen2VL fast processor 报错！
    #     model_inputs = processor(
    #         text=texts,
    #         images=examples.get("image"),
    #         truncation=True,
    #         max_length=script_args.max_seq_length,
    #         padding="max_length",
    #         return_tensors="pt",  # 这里必须 pt！！！
    #     )

    #     # 只保留文本部分，图像特征不存入数据集（关键！）
    #     input_ids = model_inputs["input_ids"].tolist()
    #     attention_mask = model_inputs["attention_mask"].tolist()

    #     # 处理答案
    #     labels = processor.tokenizer(
    #         examples["solution"],
    #         truncation=True,
    #         max_length=script_args.max_seq_length,
    #         padding="max_length",
    #     ).input_ids

    #     # 拼接 & 构造 label（prompt 部分 = -100）
    #     new_input_ids = []
    #     new_attention_mask = []
    #     new_labels = []

    #     for i in range(len(texts)):
    #         prompt_len = len(input_ids[i])
    #         input_id = input_ids[i] + labels[i]
    #         attn_mask = [1] * len(input_id)
    #         label = [-100] * prompt_len + labels[i]

    #         # 截断
    #         input_id = input_id[:script_args.max_seq_length]
    #         attn_mask = attn_mask[:script_args.max_seq_length]
    #         label = label[:script_args.max_seq_length]

    #         # 填充
    #         pad_len = script_args.max_seq_length - len(input_id)
    #         input_id += [processor.tokenizer.pad_token_id] * pad_len
    #         attn_mask += [0] * pad_len
    #         label += [-100] * pad_len

    #         new_input_ids.append(input_id)
    #         new_attention_mask.append(attn_mask)
    #         new_labels.append(label)

    #     return {
    #         "input_ids": new_input_ids,
    #         "attention_mask": new_attention_mask,
    #         "labels": new_labels,
    #     }
    def tokenize_fn(examples):
        # 1. 构造完整对话
        conversations = []
        for prompt, solution in zip(examples["prompt"], examples["solution"]):
            conv = prompt + [{"role": "assistant", "content": solution}]
            conversations.append(conv)

        # 2. 转成字符串
        texts = [processor.apply_chat_template(conv, tokenize=False) for conv in conversations]

        # 3. 处理
        model_inputs = processor(
            text=texts,
            images=examples.get("image"),
            truncation=True,
            max_length=script_args.max_seq_length,
            padding="max_length",
            return_tensors="pt",
        )

        # 4. 构造 label
        labels = model_inputs["input_ids"].clone()
        labels[model_inputs["attention_mask"] == 0] = -100
        model_inputs["labels"] = labels

 
        return {
            "input_ids": model_inputs["input_ids"],
            "attention_mask": model_inputs["attention_mask"],
            "labels": model_inputs["labels"],
        }


    # ===================== 最终 map 配置 =====================
    tokenized_data = dataset.map(
        tokenize_fn,
        batched=True,
        batch_size=2,
        num_proc=1,          # 图像必须=1
        remove_columns=dataset["train"].column_names,
    )
    train_ds = tokenized_data[script_args.dataset_train_split].select(range(30))
    print("=" * 50)
    print(f"训练集总条数：{len(train_ds)}")
    print("=" * 50)

    # 训练
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,#tokenized_data[script_args.dataset_train_split],
        eval_dataset=tokenized_data[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        data_collator=default_data_collator,
        tokenizer=processor.tokenizer,
    )

    trainer.train()
    trainer.save_model(training_args.output_dir)

if __name__ == "__main__":
    parser = TrlParser((SFTScriptArguments, TrainingArguments, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)