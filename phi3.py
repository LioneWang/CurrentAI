# -*- coding = utf-8 -*-
# @Time : 2025/3/2 18:59
# @Author : 王加炜
# @File : inference_phi3.py
# @Software : PyCharm
# Load model directly
# import os
# os.environ['HF_HOME'] = '/root/autodl-tmp/cache/'
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login

# 输入你的 Hugging Face 账户 Token，这里把自己的刚刚创建的token粘贴进来就行啦
login("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")
# 加载 Hugging Face 数据集
dataset = load_dataset("LioneWang/Reddit-5K-gpt")

# 初始化模型和分词器
tokenizer = AutoTokenizer.from_pretrained("./microsoft/Phi-3.5-mini-instruct", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("./microsoft/Phi-3.5-mini-instruct", trust_remote_code=True,torch_dtype="auto",
                                             device_map="cuda")

# 将模型移动到 GPU（如果可用）
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# 用于存储所有生成的输出
all_outputs = []

# 遍历数据集中的每条数据
for data in dataset["train"]:  # 假设数据集在 "train" 分割中
    # 提取 conversations 列
    conversations = data["conversations"]

    # 找到 human 部分的 value 作为 prompt
    human_message = next(item for item in conversations if item["from"] == "human")
    prompt = human_message["value"]

    # 将 prompt 转换为模型输入
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # 使用模型生成输出
    output = model.generate(
        **inputs,
        max_new_tokens=128,  # 控制生成的最大长度
        do_sample=True # 是否使用采样
       
    )

    # 将生成的输出解码为文本
    output_text = tokenizer.decode(output[0], skip_special_tokens=True)

    # 将输出保存到列表中
    all_outputs.append(output_text)

# 打印所有生成的输出
for i, output in enumerate(all_outputs):
    print(f"Output {i + 1}: {output}")

# 将输出保存到文件（可选）
with open("reddit_outputs.txt", "w") as f:
    for i, output in enumerate(all_outputs):
        f.write(f"Output {i + 1}: {output}\n")