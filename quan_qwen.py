import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
import os

# 设置目标文件夹路径
output_dir = './quantized_models'
os.makedirs(output_dir, exist_ok=True)  # 如果文件夹不存在，创建文件夹

# 加载预训练模型
model_name_or_path = 'qwen2.5-14b-agent-10000'  # 使用你自己的模型名称
model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

# 设置模型为评估模式
model.eval()

# 1. 动态量化
# 动态量化主要是对模型的权重进行量化，不涉及激活的量化。
quantized_model = torch.quantization.quantize_dynamic(
    model, 
    {torch.nn.Linear},  # 仅对 Linear 层进行量化
    dtype=torch.qint8  # 使用 int8 量化
)

# 2. 保存量化后的模型
quantized_model_path = os.path.join(output_dir, 'quantized_qwen_model.pth')
torch.save(quantized_model.state_dict(), quantized_model_path)

print(f"量化后的模型已保存到 {quantized_model_path}")

# 3. 测试量化模型的推理性能
input_text = "你好，今天的天气怎么样？"
inputs = tokenizer(input_text, return_tensors="pt")
input_ids = inputs["input_ids"]

# 测量推理时间
start = time.time()
with torch.no_grad():
    output_ids = quantized_model.generate(input_ids, max_new_tokens=50)
    response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
end = time.time()

print("生成的响应:", response)
print(f"推理时间: {end - start:.4f} 秒")

