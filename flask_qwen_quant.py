from fastapi import FastAPI, Request
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import time
import json
import os
from torch.quantization import convert, prepare
from accelerate import Accelerator

# 初始化 FastAPI 应用
app = FastAPI()

# 设置模型和文件夹路径
model_path = './quantized_models/quantized_qwen_model.pth'
model_name_or_path = 'qwen2.5-14b-agent-10000'  # 原始模型名称（如果需要）

# 加载量化后的模型和分词器
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
model = AutoModelForCausalLM.from_pretrained(model_name_or_path)

# 加载量化后的模型
model.load_state_dict(torch.load(model_path))

# 使用 Accelerate 进行模型并行处理（如果有多GPU等环境）
accelerator = Accelerator()
model = accelerator.prepare(model)

# 设置模型为评估模式
model.eval()

# 用于清理内存缓存
def torch_gc():
    torch.cuda.empty_cache()

# 处理 Spark 风格对话文本
def transform_spark_to_qwen(text):
    total_lst = []
    text = text.replace("<Bot>", "").strip()
    lst = [t.strip() for t in text.split("<end>") if len(t.replace("<ret>", "\n").strip()) > 0]

    def transform_text(text):
        text = text.strip()
        text = text.replace("<end>", "").replace("<Bot>", "").replace("<User>", "").strip()
        text = text.replace("<ret>", "\n").strip()
        return text

    for i, e in enumerate(lst):
        e = transform_text(e)
        if i == 0 and e.startswith("<System>"):
            total_lst.append({"role": "system", "content": e})
        elif i % 2 == 1:
            total_lst.append({"role": "user", "content": e})
        elif i > 0 and i % 2 == 0:
            total_lst.append({"role": "assistant", "content": e})
        else:
            continue
    return total_lst


@app.post("/query/")
async def qwen_query(request: Request):
    # 获取请求内容
    json_post_raw = await request.json()
    json_post_dict = json.loads(json.dumps(json_post_raw))
    messages = json_post_dict.get('dialogue')

    # 如果输入的是字符串而不是列表，进行转换
    if isinstance(messages, str):
        messages = transform_spark_to_qwen(messages)

    # 使用 tokenizer 处理输入
    input_ids = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([input_ids], return_tensors="pt").to(model.device)
    attention_mask = model_inputs['attention_mask'].to(model.device)
    input_ids = model_inputs['input_ids'].to(model.device)

    start = time.time()

    # 执行推理，生成响应
    generated_ids = model.generate(input_ids, attention_mask=attention_mask, max_new_tokens=50)

    # 获取生成的文本
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    # 计算推理时间
    time_diff = time.time() - start

    # 清理内存
    torch_gc()

    # 返回结果
    return {
        "response": response,
        "status": 200,
        "time": time_diff,
    }

# 启动服务
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8009)

