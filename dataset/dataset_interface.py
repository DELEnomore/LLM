from abc import abstractmethod

from datasets import load_dataset

from configs import CACHE_DIR, DATASET_CACHE_DIR


class DatasetInterface:
    PATH = ''

    NAME = None

    def __init__(self, tokenizer):
        self.data = load_dataset(self.PATH, self.NAME, cache_dir=DATASET_CACHE_DIR)
        self.tokenizer = tokenizer

    def get_data(self):
        return self.data

    @abstractmethod
    def tokenize_function(self, tokenize):
        pass

    @abstractmethod
    def build_prompt(input) -> str:
        pass