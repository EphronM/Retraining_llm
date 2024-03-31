
# Fine-Tuning Llama2 Model using QLoRA in Google Colab

## Overview
This project involves retraining the Llama2 generative model on a custom dataset using Google Colab. The dataset used is called openassistant-guanaco, containing around 1,000 data points, which was preprocessed and fine-tuned on the Llama2 model using parameter-efficient fine-tuning (PEFT) techniques like QLoRA to drastically reduce VRAM usage. The fine-tuned model was then uploaded to the Hugging Face model repository for accessibility.


## Dataset and Model Information



* **Dataset**: openassistant-guanaco
* **Preprocessed Dataset**: EphronM/guanaco-llama2-1k
* **Fine-Tuned Model ID**: EphronM/Llama-2-7b-chat-finetune

Data and model is available in Hugging face 


## Fine-Tuning Process

* **Preprocessing**: The openassistant-guanaco dataset containing 1,000 data points was preprocessed to prepare it for fine-tuning.

* **Fine-Tuning Technique**: Parameter-efficient fine-tuning (PEFT) techniques like QLoRA were employed to fine-tune the Llama2 model efficiently while drastically reducing VRAM usage.

* **Fine-Tuning in Google Colab**: Google Colab was used for fine-tuning the model due to its free access to GPU resources and easy integration with Hugging Face libraries.

* **Model Evaluation**: The fine-tuned model's performance was manually evaluated on its effectiveness in generating responses.

## Model Usage
To use the fine-tuned Llama2 model for generative chat tasks:

1. **Hugging Face Model Hub**: The fine-tuned model is available on the Hugging Face Model Hub.

2. **Model Installation**: Install the model using the Hugging Face Transformers library:

```
pip install transformers
```
3. **Model Loading**: Load the model using its identifier:
```
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("EphronM/Llama-2-7b-chat-finetune")
tokenizer = AutoTokenizer.from_pretrained("EphronM/Llama-2-7b-chat-finetune")
```
4. **Generating Responses**: Use the loaded model and tokenizer to generate responses to prompts:
```
prompt = "User: Hello! How are you?"
input_ids = tokenizer.encode(prompt, return_tensors="pt")
output = model.generate(input_ids, max_length=50, pad_token_id=tokenizer.eos_token_id)
response = tokenizer.decode(output[0], skip_special_tokens=True)
print("Bot:", response)
```

Fin .