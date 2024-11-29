import os
import torch
import torch.quantization
from transformers import AutoModelForCausalLM, AutoTokenizer
import time

# 设置目标文件夹路径
output_dir = './quantized_models'
os.makedirs(output_dir, exist_ok=True)  # 如果文件夹不存在，创建文件夹

# 加载预训练模型
model_name_or_path = 'qwen2.5-14b-agent-10000'
model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

# 设置模型为评估模式
model.eval()

# 1. 设置量化配置
# 使用 fbgemm 后端进行量化，适用于 CPU
model.qconfig = torch.quantization.get_default_qconfig('fbgemm')  # 选择量化配置
torch.quantization.prepare(model, inplace=True)  # 为量化做好准备

# 2. 校准模型（需要提供代表性数据）
# 假设我们使用一些简单的输入数据进行校准
calibration_data = [
    "你好，今天的天气怎么样？",
    "请帮我推荐一本好书。",
    "你觉得明天会下雨吗？"
]

# 校准过程：将模型在一些输入数据上运行，以便量化激活值
for text in calibration_data:
    inputs = tokenizer(text, return_tensors="pt")
    input_ids = inputs["input_ids"]
    with torch.no_grad():
        model(input_ids)

# 3. 转换为量化模型
quantized_model = torch.quantization.convert(model, inplace=True)

# 4. 保存量化后的模型到指定文件夹
quantized_model_path = os.path.join(output_dir, 'quantized_qwen_model.pth')
torch.save(quantized_model.state_dict(), quantized_model_path)

print(f"量化后的模型已保存到 {quantized_model_path}")

# 测试量化模型的推理
input_text = "你好，今天的天气怎么样？"
inputs = tokenizer(input_text, return_tensors="pt")
input_ids = inputs["input_ids"]

start = time.time()
with torch.no_grad():
    output_ids = quantized_model.generate(input_ids, max_new_tokens=50)
    response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
end = time.time()

print("生成的响应:", response)
print(f"推理时间: {end - start:.4f} 秒")

