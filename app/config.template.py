# SM Technology Chatbot configuration template
# Copy this to config.py and add your own token

# Hugging Face token - Get this from huggingface.co/settings/tokens
HUGGINGFACE_TOKEN = ""

# Model settings
MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Dataset settings
DATASET_PATH = "data/dataset.json"

# Generation parameters
MAX_LENGTH = 400
TEMPERATURE = 0.6
TOP_P = 0.92
TOP_K = 40
REPETITION_PENALTY = 1.1