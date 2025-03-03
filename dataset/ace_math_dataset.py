from datasets import load_dataset

from dataset.dataset_interface import DatasetInterface


class AceMathDataset(DatasetInterface):


    DATASET_NAME = 'nvidia/AceMath-Instruct-Training-Data'

    def tokenize_function(examples, tokenizer):
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
            max_length=512,
            truncation=True,
            padding="max_length",
            return_tensors="pt"  # 根据训练框架调整（如Trainer可省略）
        )

        # 4. 对response分词并设置为labels
        labels = tokenizer(
            responses,
            max_length=512,
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

    def build_prompt(input) -> str:
        pass