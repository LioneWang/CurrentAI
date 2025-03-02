# 导入必要的库
from unsloth import FastLanguageModel
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed, TextStreamer
from datasets import load_dataset
from huggingface_hub import login

# 设置 Hugging Face 登录 Token
login("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")  # 替换为你的 Hugging Face Token

# 加载 Hugging Face 数据集
dataset = load_dataset("LioneWang/Reddit-5K-gpt")

# 设置模型参数
max_seq_length = 18000  # 上下文长度
dtype = None  # 自动检测数据类型
load_in_4bit = True  # 使用 4bit 量化以减少内存占用

# 初始化模型和分词器
if True:
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="LioneWang/Finetuned-Phi3.5-LORA",  # 替换为你的模型名称
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
    )
    FastLanguageModel.for_inference(model)  # 启用推理优化

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

    # 构造 messages 格式
    messages = [{"from": "human", "value": prompt}]

    # 将 messages 转换为模型输入
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,  # 必须添加生成提示
        return_tensors="pt",
    ).to(device)

    # 使用模型生成输出
    text_streamer = TextStreamer(tokenizer, skip_prompt=True)
    output = model.generate(
        input_ids=inputs,
        streamer=text_streamer,
        max_new_tokens=128,  # 控制生成的最大长度
        use_cache=True,  # 使用缓存加速
    )

    # 将生成的输出解码为文本
    output_text = tokenizer.decode(output[0], skip_special_tokens=True)

    # 将输出保存到列表中
    all_outputs.append(output_text)



# 将输出保存到文件（可选）
with open("reddit_outputs.txt", "w") as f:
    for i, output in enumerate(all_outputs):
        f.write(f"Output {i + 1}: {output}\n")