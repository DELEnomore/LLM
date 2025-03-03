import os.path
import time

import pandas as pd
import pyarrow.parquet as pq
import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import LlamaForCausalLM, LlamaTokenizer, Trainer, TrainingArguments, AutoModelForCausalLM, \
    AutoTokenizer

from configs import MODEL_NAME
from dataset.ace_math_dataset import AceMathDataset


def fine_tune():
    # 检查并添加填充标记
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        load_in_4bit=True,  # 4-bit 模式
        torch_dtype=torch.float16,  # 混合精度
        device_map="auto",  # 自动分配到 GPU
    )

    lora_config = LoraConfig(
        r=8,  # LoRA 的秩
        lora_alpha=32,  # LoRA 的缩放因子
        lora_dropout=0.05,  # Dropout 概率
        bias="none",  # LoRA bias 设置
        task_type="CAUSAL_LM",  # 任务类型：自回归文本生成
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        # 如果需要根据具体模型结构，调整 target_modules
    )

    peft_model = get_peft_model(model, lora_config)
    peft_model.print_trainable_parameters()  # 查看可训练参数量

    dataset_instance = AceMathDataset()
    data = dataset_instance.get_data()
    # 1. 拆分训练集和验证集
    split_data = data["train"].train_test_split(
        test_size=0.1,
        seed=42
    )
    train_data = split_data["train"]
    valid_data = split_data["test"]

    train_tokenized = train_data.map(dataset_instance.tokenize_function, batched=True)
    valid_tokenized = valid_data.map(dataset_instance.tokenize_function, batched=True)

    training_args = TrainingArguments(
        output_dir=MODEL_OUTPUT_DIR,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=8,
        num_train_epochs=10,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=50,
        evaluation_strategy="steps",
        eval_steps=200,
        save_strategy="steps",
        save_steps=200,
    )

    trainer = Trainer(
        model=peft_model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=valid_tokenized,
    )

    trainer.train(
        # resume_from_checkpoint=True
    )
    trainer.save_model()
    pass

def run_fine_tuned_model():
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,  # Use float16 for faster performance on compatible GPUs
        device_map="auto",
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

            # 构建 Promp
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


def main():

    fine_tune()
    run_fine_tuned_model()


# 微调模型
if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    MODEL_OUTPUT_DIR = os.path.join(os.environ['TRANSFORMERS_CACHE'], MODEL_NAME + '-finetuned')
    main()
