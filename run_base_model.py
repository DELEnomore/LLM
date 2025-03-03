import os
import time

import transformers
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from configs import MODEL_NAME, CACHE_DIR


def main():
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

    # 3. 交互式推理循环
    while True:
        try:
            # 用户输入
            inst = input("\nUser: ").strip()


            start_time = time.time()
            outputs = text_generator(
                inst,
                do_sample=True,
                top_p=0.9,  # 平衡多样性与质量
                temperature=0.7,  # 降低随机性
                max_new_tokens=256,  # 控制生成长度（不是总长度！）
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
            generation_time = time.time() - start_time

            generated_text = outputs[0]["generated_text"]
            print(f"\nModel: {generated_text}")
            print(f"[生成耗时: {generation_time:.2f}s]")

        except KeyboardInterrupt:
            print("\n退出交互。")
            break


main()
