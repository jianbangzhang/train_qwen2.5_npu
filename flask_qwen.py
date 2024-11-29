from fastapi import FastAPI, Request
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import uvicorn
import json
import torch
import torch_npu
import time
from accelerate import Accelerator


def torch_gc():
    if torch.npu.is_available():
        with torch.npu.device(model.device):
            torch.npu.empty_cache()


def transform_spark_to_qwen(text):
    total_lst=[]
    text=text.replace("<Bot>","").strip()
    lst = [t.strip() for t in text.split("<end>") if len(t.replace("<ret>","\n").strip())>0]

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


accelerator = Accelerator()
app = FastAPI()


@app.get("/ping")
async def ping():
    return {"message":"pong"}




@app.post("/query/")
async def qwen_query(request: Request):
    global model, tokenizer
    json_post_raw = await request.json()
    json_post = json.dumps(json_post_raw)
    json_post_dict = json.loads(json_post)
    messages = json_post_dict.get('dialogue')

    if not isinstance(messages, list) and isinstance(messages, str):
        print("raw:\n",messages)
        messages = transform_spark_to_qwen(messages)

    print("input:\n", messages)
    input_ids = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([input_ids], return_tensors="pt").to(model.device)
    attention_mask = model_inputs['attention_mask'].to(model.device)
    input_ids = model_inputs['input_ids'].to(model.device)

    start = time.time()
    generated_ids = model.generate(input_ids, attention_mask=attention_mask, max_new_tokens=max_model_len)
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    time_diff = time.time()-start
    answer = {
        "response": response,
        "status": 200,
        "time": time_diff,
    }
    print('response:\n' + repr(response))
    print("time:\n",time_diff)
    torch_gc()
    return answer



if __name__ == '__main__':
    max_model_len=4096
    model_name_or_path = 'qwen2.5-14b-agent-10000'

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype="auto",
        device_map="auto",
        offload_folder=None
    )

    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model, tokenizer = accelerator.prepare(model, tokenizer)

    uvicorn.run(app, host='0.0.0.0', port=8009)
