import pandas as pd
import pyarrow.parquet as pq
import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import LlamaForCausalLM, LlamaTokenizer, Trainer, TrainingArguments, AutoModelForCausalLM, \
    AutoTokenizer


def main():
    # Load the model path
    model_path = "D:\\Code\\LLM\\Llama-3.2-3B-Instruct"
    # Load the tokenizer and model with GPU support
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_path)

    tokenizer.pad_token = tokenizer.eos_token
    # 检查并添加填充标记

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
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

    dataset = load_dataset("hfl/ruozhiba_gpt4")
    # 1. 拆分训练集和验证集
    split_dataset = dataset["train"].train_test_split(
        test_size=0.1,
        seed=42
    )
    train_data = split_dataset["train"]
    valid_data = split_dataset["test"]


    def tokenize_function(examples, max_length=512):
        # 1. 构建模型输入（Prompt）
        prompts = []
        for inst, inp in zip(examples['instruction'], examples['input']):
            if inp.strip():  # 如果input不为空
                prompt = f"### Instruction:\n{inst}\n\n### Input:\n{inp}\n\n### Response:\n"
            else:  # 如果input为空
                prompt = f"### Instruction:\n{inst}\n\n### Response:\n"
            prompts.append(prompt)

        # 2. 构建标签（仅包含output部分）
        responses = examples['output']

        # 3. 对prompt和response分别分词
        model_inputs = tokenizer(
            prompts,
            max_length=max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"  # 根据训练框架调整（如Trainer可省略）
        )

        # 4. 对response分词并设置为labels
        labels = tokenizer(
            responses,
            max_length=max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"  # 根据训练框架调整
        )["input_ids"]

        # 5. 处理填充token的损失掩码
        # 将padding部分的label标记为-100（损失函数自动忽略）
        labels[labels == tokenizer.pad_token_id] = -100

        # 6. 将labels添加到输入字典
        model_inputs["labels"] = labels
        return model_inputs

    train_tokenized = train_data.map(tokenize_function, batched=True)
    valid_tokenized = valid_data.map(tokenize_function, batched=True)

    training_args = TrainingArguments(
        output_dir="./lora-llama-ckpt",
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


# 微调模型
if __name__ == "__main__":
    main()
