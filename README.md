# Python setup
```
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

# Download Llama
Download LLM model from HuggingFace.<br>
[Llama-3-ELYZA-JP-8B](https://huggingface.co/elyza/Llama-3-ELYZA-JP-8B)

# Run
```
.\.venv\Scripts\activate
python vba_backend.py --model_path "Path to model"
```
