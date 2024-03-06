# -*- coding: utf-8 -*-

"""Llama_2_PEFT_QLoRA.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1xhO3vxluFqUe5RPPvZhbxVfC1cTVPYgb

# Llama 2 Instruction Fine-Tuning Using PEFT and QLoRA
**USE CASE: instruction fine-tune LLM for a specific downstream task**.

Dependencies: pip install accelerate peft bitsandbytes transformers trl
"""

import warnings
warnings.filterwarnings("ignore")

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from trl import SFTTrainer
#import deepspeed


# functions to build prompts for the LLM being fine-tuned
def build_prompt_with_input(sample: dict) -> str:
  """
    Construct a prompt string with both instruction and input from a sample.
    Parameters:
    - sample (dict): A dictionary containing 'instruction', 'input', and 'output' keys.
    Returns:
    - str: A formatted string including the instruction, input, and expected output.
  """
  return f"""### Instruction:
{sample['instruction']}: {sample['input']}

### Response:
{sample['output']}
"""


def build_prompt_no_input(sample: dict) -> str:
  """
    Construct a prompt string with instruction only from a sample.
    Parameters:
    - sample (dict): A dictionary containing 'instruction' and 'output' keys.
    Returns:
    - str: A formatted string including the instruction and expected output.
  """
  return f"""### Instruction:
{sample['instruction']}

### Response:
{sample['output']}
"""


def formatting_func(sample: dict) -> str:
  """
    Determine the appropriate prompt format based on the presence of input field in the sample.
    Parameters:
    - sample (dict): A dictionary containing 'instruction', 'input', and 'output' keys.
    Returns:
    - str: A formatted prompt string built based on whether 'input' is empty or not.
  """
  return build_prompt_no_input(sample) if sample["input"] == "" else build_prompt_with_input(sample)


# load training datasets from HuggingFace hub
dataset_alpaca = load_dataset("vicgalle/alpaca-gpt4", split="train[:1000]")

# Base Llama 2 13b model to be fine-tuned
base_model_id = "NousResearch/Llama-2-13b-chat-hf"
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)
model = AutoModelForCausalLM.from_pretrained(base_model_id, quantization_config=quantization_config)

# load tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    base_model_id,
    model_max_length=512,
    padding_side="left",
    add_eos_token=True)
tokenizer.pad_token = tokenizer.eos_token

# prepare model for fine-tuning
model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

# low-rank adaptation (LORA) parameters
config = LoraConfig(
    r=8,
    lora_alpha=8,
    target_modules=[ "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "lm_head", ],
    bias="none",
    lora_dropout=0.05,  # Conventional
    task_type="CAUSAL_LM",
)

# prepare model for parameter efficient fine-tuning (PEFT)
model = get_peft_model(model, config)

# set training arguments
YOUR_HF_USERNAME = "agnedil"
new_model_id = "llama-2-13b-alpaca-gpt4"
output_dir = f"{YOUR_HF_USERNAME}/{new_model_id}"
per_device_train_batch_size = 2
gradient_accumulation_steps = 4
optim = "paged_adamw_8bit"
save_steps = 10
logging_steps = 10
learning_rate = 2e-4
max_grad_norm = 0.3
max_steps = 500
warmup_ratio = 0.03
lr_scheduler_type = "constant"

# set training arguments
training_arguments = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    optim=optim,
    save_steps=save_steps,
    logging_steps=logging_steps,
    learning_rate=learning_rate,
    max_grad_norm=max_grad_norm,
    max_steps=max_steps,
    warmup_ratio=warmup_ratio,
    lr_scheduler_type=lr_scheduler_type,
    deespeed="ds_config.json",
    gradient_checkpointing=True,
    push_to_hub=True,
)

# set SFTTrainer object
trainer = SFTTrainer(
    model=model,
    args=training_arguments,
    train_dataset=dataset_alpaca,
    packing=True,
    tokenizer=tokenizer,
    max_seq_length=512,
    formatting_func=formatting_func,
)

# conduct the actual fine-tuning; the fine-tuned model is pushed to HuggingFace hub
trainer.train()

# save model and tokenizer
trainer.model.save_pretrained(new_model_id)
trainer.tokenizer.save_pretrained(new_model_id)