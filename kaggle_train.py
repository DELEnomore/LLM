"""
日志配置，建议放到最前面
"""
import logging
import os
import sys
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from huggingface_hub import login
import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import LlamaTokenizer, Trainer, TrainingArguments


def config_logger():
    # 获取默认的根 logger
    root_logger = logging.getLogger()
    # 设置日志级别
    root_logger.setLevel(logging.INFO)
    # 创建控制台 Handler，输出到控制台
    console_handler = logging.StreamHandler()
    # 创建日志格式器
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    # 将 Handler 添加到根 logger
    root_logger.addHandler(console_handler)
    # 进行日志记录
    logging.info('logging output test')




"""
一些全局的变量
"""


"""
加载模型
"""


# 登录hugging face
# login("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")





def main():
    os.environ['WANDB_DISABLED'] = 'true'

    output_dir = "/kaggle/working/llama_finetuned_model"

    config_logger()
    # model_name = "Qwen/Qwen2.5-1.5B"
    # model_name = "meta-llama/Llama-3.2-3B"
    model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir='.cache'
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 量化配置
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,  # 启用 4-bit 量化
        bnb_4bit_quant_type="nf4",  # 量化数据类型（推荐 nf4）
        bnb_4bit_use_double_quant=True,  # 启用双量化（进一步压缩显存）
        bnb_4bit_compute_dtype=torch.float16  # 计算时使用 bfloat16 加速
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        # quantization_config=bnb_config,
        cache_dir='.cache',
        torch_dtype=torch.float16,  # 混合精度
        device_map="auto",  # 自动分配到 GPU
    )
    logging.info('model loaded')

    # 将模型移至 GPU（如果可用）
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info('cuda available: %s', torch.cuda.is_available())
    logging.info(f'Model is on device: {model.device}')
    """
    读取数据集，定义tokenize_function
    """
    dataset = load_dataset("hfl/ruozhiba_gpt4")

    # 数据预处理函数
    def tokenize_function(example):
        # 将instruction和output合并为一个输入
        input = example['instruction']
        target = example['output']
        # 将 'instruction' 和 'output' 作为序列对进行编码
        model_input = tokenizer(input, padding="max_length", truncation=True, max_length=512)
        # 对目标输出进行编码
        labels = tokenizer(target, padding="max_length", truncation=True, max_length=512)
        labels["input_ids"] = [[-100 if token == tokenizer.pad_token_id else token for token in token_list] for
                               token_list in
                               labels["input_ids"]]
        # 将标签放入模型输入字典中
        model_input["labels"] = labels["input_ids"]
        return model_input

    """
    对数据集进行tokenize转换
    """
    # 进行预处理
    tokenized_dataset = dataset.map(tokenize_function, num_proc=4, batched=True)
    # 划分数据集为训练集和验证集（比如 80% 训练集，20% 验证集）
    splited_dataset = tokenized_dataset['train'].train_test_split(test_size=0.2)
    train_dataset = splited_dataset['train']
    val_dataset = splited_dataset['test']
    # 确保数据集是有效的 Dataset 对象
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    """
    LoRA模型配置
    """
    lora_config = LoraConfig(
        r=8,  # 低秩适应的秩
        lora_alpha=32,  # 缩放系数
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    peft_model = get_peft_model(model, lora_config)  # 应用LoRA
    model.print_trainable_parameters()
    """
    展开训练
    """
    # 设置训练参数
    training_args = TrainingArguments(
        output_dir=output_dir + "/results",  # 输出目录
        learning_rate=3e-4,
        num_train_epochs=3,  # 训练周期数
        per_device_train_batch_size=2,  # 每设备的batch size
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=8,
        logging_dir=output_dir + "/logs",  # 日志目录
        logging_steps=50,  # 日志输出频率
        eval_steps=200,
        save_steps=500,  # 保存模型的频率
        save_total_limit=3,  # 最大保存的模型数量
        disable_tqdm=False,
        report_to=None,
        dataloader_num_workers=4,
        gradient_checkpointing=True,
        fp16=True,
    )
    # 创建Trainer实例
    trainer = Trainer(
        model=peft_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,  # 如果有验证集的话
        tokenizer=tokenizer,
    )
    logging.info('now start training')
    # 训练模型
    trainer.train()
    # 保存模型和中间结果到本地
    model_save_path = output_dir + "final_result"
    model.save_pretrained(model_save_path)
    logging.info('train done, model saved')


if __name__ == '__main__':
    main()