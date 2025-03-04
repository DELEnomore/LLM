from dataset.dataset_interface import DatasetInterface


class RuozhiDataset(DatasetInterface):

    PATH = 'hfl/ruozhiba_gpt4'

    def tokenize_function(self, example):
        # 将instruction和output合并为一个输入
        input = example['instruction']
        target = example['output']
        # 将 'instruction' 和 'output' 作为序列对进行编码
        model_input = self.tokenizer(input, padding="max_length", truncation=True, max_length=512)
        # 对目标输出进行编码
        labels = self.tokenizer(target, padding="max_length", truncation=True, max_length=512)
        labels["input_ids"] = [[-100 if token == self.tokenizer.pad_token_id else token for token in token_list] for
                               token_list in
                               labels["input_ids"]]
        # 将标签放入模型输入字典中
        model_input["labels"] = labels["input_ids"]
        return model_input

    def build_prompt(input) -> str:
        pass