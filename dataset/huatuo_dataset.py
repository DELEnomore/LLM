from dataset.dataset_interface import DatasetInterface


class HuatuoDataset(DatasetInterface):

    PATH = "FreedomIntelligence/HuatuoGPT2-Pretraining-Instruction"

    NAME = "Meidcal_Encyclopedia_cn"

    def tokenize_function(self, examples):
        inputs = []
        labels = []
        # print(f'examples: {examples.keys()}')
        for conversation in examples['conversations']:
            # 假设每个对话的第一部分是 human 提问，第二部分是 gpt 的回答
            dialog_input = ""
            dialog_output = ""

            # 遍历每个对话，区分 human 和 gpt
            for turn in conversation:
                if turn["from"] == "human":
                    dialog_input = turn["value"]  # 用户的提问作为输入
                elif turn["from"] == "gpt":
                    dialog_output = turn["value"]  # GPT 的回答作为标签

            # 将输入和标签添加到列表中
            inputs.append(dialog_input.strip())  # 用户提问
            labels.append(dialog_output.strip())  # GPT 回答

        # 使用分词器将输入和输出数据转换为模型输入
        model_inputs = self.tokenizer(inputs, padding='max_length', max_length=512, truncation=True)
        labels_tokenized = self.tokenizer(labels, padding='max_length', max_length=512, truncation=True)

        # 将标签添加到模型输入
        model_inputs["labels"] = labels_tokenized["input_ids"]

        return model_inputs

    def build_prompt(input) -> str:
        pass