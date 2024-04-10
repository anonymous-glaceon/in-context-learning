# Code for "Larger language models do in-context learning differently."

Pipeline:
1. Download data from HuggingFace using `create_huggingface_datasets.py`
2. Compile data into prompts using `load_huggingface_datasets.py`
3. Run OpenAI models on prompts using `call_openai.py`
