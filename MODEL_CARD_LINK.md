# Trained compression model

The fine-tuned LLMLingua2 token-classification model is stored on Hugging Face Hub:

https://huggingface.co/MaxOn14/llmlingua2-qwen-finetuned

Use it with:

```python
from transformers import AutoModelForTokenClassification, AutoTokenizer

model_id = "MaxOn14/llmlingua2-qwen-finetuned"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForTokenClassification.from_pretrained(model_id)
