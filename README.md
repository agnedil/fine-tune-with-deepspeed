# Fine-Tuning Llama 13B with HuggingFace Transformers and DeepSpeed

This README provides instructions for fine-tuning the Llama 13B model on the "alpaca-gpt-4" dataset using HuggingFace Transformers and DeepSpeed.

## Prerequisites

- Python 3.6 or later.
- Basic familiarity with Python programming and virtual environments.

## Setup

### 1. Create a Virtual Environment (Optional)

Creating a virtual environment is recommended to avoid conflicts with system-wide packages.

```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

### 2. Check if you have access to GPU
```bash
nvcc --version
```

### 3. If you need access to HuggingFace Hub
If you need access to HuggingFace Hub, provide the access token after running this commend:
```bash
huggingface-cli login
```

### 4. Install Dependencies
Install the required libraries using pip:

```bash
pip install accelerate peft bitsandbytes transformers trl datasets deepspeed
```
If you encounter a matrix multiplication error, downgrade the transformers package according to [this issue](https://discuss.huggingface.co/t/help-with-llama-2-finetuning-setup/50035/4):
```bash
!pip install git+https://github.com/huggingface/transformers@v4.31-release
```

### 5. Run the Fine-Tuning Script On One GPU
The fine-tuning script, script.py, is prepared for execution along with a config file ds_config.json. To run the script with DeepSpeed, execute the following command in your terminal in the same directory as the above two files (or adjust file paths respectively):

```bash
deepspeed --num_nodes 1 --num_gpus=1 script.py
```