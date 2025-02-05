import os
import time

import transformers
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from configs import MODEL_NAME, CACHE_DIR



tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=MODEL_NAME, cache_dir=CACHE_DIR)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,  # Use float16 for faster performance on compatible GPUs
    device_map="auto",
    cache_dir=CACHE_DIR
)

# Initialize pipeline with GPU
text_generator = transformers.pipeline(
    task="text-generation",
    model=model,
    tokenizer=tokenizer,
    truncation=True,  # 必须启用截断
    padding="max_length",  # 与训练时一致
    max_length=512,  # 必须与训练时 max_length 一致
)


def build_prompt(instruction, input_text=None) -> str:
    if input_text and input_text.strip():
        return f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n"
    else:
        return f"### Instruction:\n{instruction}\n\n### Response:\n"


def extract_response(full_text: str) -> str:
    """从生成的完整文本中提取 Response 部分"""
    response_start = full_text.find("### Response:\n")
    if response_start == -1:
        return "[ERROR] Response separator not found."
    response = full_text[response_start + len("### Response:\n"):].strip()
    # 安全截断：防止模型重复生成模板
    return response.split("### Instruction:")[0].split("### Response:")[0].strip()


# 3. 交互式推理循环
while True:
    try:
        # 用户输入
        inst = input("\nUser: ").strip()
        if not inst:
            print("请输入有效的指令！")
            continue

        # 构建 Prompt
        prompt = build_prompt(instruction=inst)

        start_time = time.time()
        outputs = text_generator(
            prompt,
            do_sample=True,
            top_p=0.9,  # 平衡多样性与质量
            temperature=0.7,  # 降低随机性
            max_new_tokens=256,  # 控制生成长度（不是总长度！）
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
        generation_time = time.time() - start_time

        generated_text = outputs[0]["generated_text"]
        response = extract_response(generated_text)
        print(f"\nModel: {response}")
        print(f"[生成耗时: {generation_time:.2f}s]")

    except KeyboardInterrupt:
        print("\n退出交互。")
        break
