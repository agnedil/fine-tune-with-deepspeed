# Fine-Tuning Llama 13B with HuggingFace Transformers and DeepSpeed

This README describes steps for instruction fine-tuning of the `Llama 13B model` on the `alpaca-gpt-4` dataset using HuggingFace Transformers and DeepSpeed. The script can be run on a single A100 GPU, for instance in Goggle Colab - just open the terminal and complete the steps listed below. Alternatively, you can run this on your choice of one or several GPUs.

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

### 2. Clone this code repository

If cloning with HTTPS:
```bash
git clone https://github.com/agnedil/fine-tune-with-deepspeed.git
```

### 3. Make sure you have access to GPU
```bash
nvcc --version
```

### 4. Ensure access to HuggingFace Hub
If you need access to HuggingFace Hub, provide the access token after running this command:
```bash
huggingface-cli login
```

### 5. Install Dependencies
Install the required libraries using pip:

```bash
pip install accelerate peft bitsandbytes transformers trl datasets deepspeed
```
If you encounter a matrix multiplication error when running the script, you may want to downgrade the transformers package according to [this issue](https://discuss.huggingface.co/t/help-with-llama-2-finetuning-setup/50035/4):
```bash
!pip install git+https://github.com/huggingface/transformers@v4.31-release
```

### 6. Run the Fine-Tuning Script on One GPU
The fine-tuning script, script.py, is prepared for execution along with a config file ds_config.json. To run the script with DeepSpeed, execute the following command in your terminal in the same directory as the above two files (or adjust file paths respectively):

```bash
deepspeed --num_nodes 1 --num_gpus=1 script.py
```

### 7. Run the Fine-Tuning Script on Multiple GPUs
Modify ds_config.json and replace m with a specific number of GPU nodes and n with a specific number of GPUs per node in the command below. For more details on running DeepSpeed, see the References section below.

```bash
deepspeed --num_nodes {m} --num_gpus={n} script.py
```

### 8. References
* [DeepSpeed AI](https://www.deepspeed.ai/)
* [Integration of DeepSpeed with HuggingFace transformers](https://huggingface.co/docs/transformers/main/deepspeed)
* [Running DeepSpeed on a single GPU](https://www.deepspeed.ai/tutorials/zero-offload/)
* [Miscrosoft DeepSpeed on GitHub](https://github.com/microsoft/DeepSpeed)