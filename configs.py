import os

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.cache')
MODEL_NAME = "D:\\Code\\LLM\\Llama-3.2-3B-Instruct"

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ['TRANSFORMERS_CACHE'] = CACHE_DIR
