import torch
import torch_npu
import torch.quantization
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch_npu
from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import Accelerator
from time import time


accelerator = Accelerator()

model_path = "../train_qwen2.5_14b/code/output_qwen14b_checkpoints_new/checkpoint-2900"
model_path0 = 'qwen2.5-agent'
model_path1 = '/workspace/train_qwen2.5_14b/qwen2.5-14b'
model_path2 = 'checkpoint-2700'
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype="auto",
    device_map="auto",
    offload_folder=None
)

model.eval()
tokenizer = AutoTokenizer.from_pretrained(model_path)
model, tokenizer = accelerator.prepare(model, tokenizer)


# 准备量化配置
model.eval()
# 设置为量化准备模式
model.qconfig = torch.quantization.get_default_qconfig('fbgemm')

# 使用代表性数据集进行统计信息收集
def prepare_model_for_quantization(model):
    # 模型设置为量化准备模式
    model = torch.quantization.prepare(model)
    return model

# 输入代表性数据（这里只是简单的文本，实际使用时可能是一个数据生成器或加载实际数据）
inputs = tokenizer("Hello, how are you?", return_tensors="pt").to(model.device)

# 前向传播并收集统计信息
with torch.no_grad():
    model(inputs['input_ids'])

# 量化转换
def convert_model_to_quantized(model):
    model = torch.quantization.convert(model)
    return model

# 量化模型
quantized_model = convert_model_to_quantized(model)

# 保存量化后的模型
torch.save(quantized_model.state_dict(), "quantized_model.pth")

# 使用量化模型进行推理
quantized_model.eval()
with torch.no_grad():
    outputs = quantized_model(inputs['input_ids'])

# 打印输出
print(outputs)
