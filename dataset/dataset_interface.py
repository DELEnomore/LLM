from abc import abstractmethod

from datasets import load_dataset


class DatasetInterface:
    DATASET_NAME = ''

    def __init__(self):
        self.data = load_dataset(self.DATASET_NAME)

    def get_data(self):
        return self.data

    @abstractmethod
    def tokenize_function(self, tokenize):
        pass

    @abstractmethod
    def build_prompt(input) -> str:
        pass