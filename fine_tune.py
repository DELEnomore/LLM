import os.path
import time

import pandas as pd
import pyarrow.parquet as pq
import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import LlamaForCausalLM, LlamaTokenizer, Trainer, TrainingArguments, AutoModelForCausalLM, \
    AutoTokenizer

from configs import MODEL_CACHE_DIR, OUTPUT_DIR
from dataset.huatuo_dataset import HuatuoDataset
from dataset.ruozhi_dataset import RuozhiDataset


def fine_tune():
    MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    MODEL_OUTPUT_DIR = os.path.join(OUTPUT_DIR, MODEL_NAME + '-finetuned')
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=MODEL_NAME, cache_dir=MODEL_CACHE_DIR)
    tokenizer.pad_token = tokenizer.eos_token

    dataset_instance = HuatuoDataset(tokenizer)
    data = dataset_instance.get_data()

    tokenized_dataset = data.map(dataset_instance.tokenize_function, num_proc=4, batched=True)
    splited_dataset = tokenized_dataset['train'].train_test_split(test_size=0.2)
    train_dataset = splited_dataset['train']
    val_dataset = splited_dataset['test']
    # 检查并添加填充标记

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,  # 混合精度
        device_map="auto",  # 自动分配到 GPU
        cache_dir=MODEL_CACHE_DIR
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
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer
    )

    trainer.train(
        # resume_from_checkpoint=True
    )
    trainer.save_model()
    pass


# 微调模型
if __name__ == "__main__":
    fine_tune()
