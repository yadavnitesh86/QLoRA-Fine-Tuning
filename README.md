# QLoRA Fine-Tuning: TinyLlama on Databricks Dolly

Fine-tuning TinyLlama-1.1B-Chat with QLoRA on the Databricks Dolly instruction dataset, followed by evaluation on held-out examples and a comparison between the base and fine-tuned models.

## Overview

This project demonstrates a parameter-efficient LLM fine-tuning workflow using QLoRA.

The base TinyLlama model is loaded in 4-bit precision using NF4 quantization and double quantization. Instead of updating the full model, LoRA adapters are trained on selected attention projection layers.

The project also includes held-out evaluation so that the fine-tuned model can be compared against the original model on unseen instructions.

## Workflow

```text
Dolly Dataset
      ↓
Select 2,000 Examples
      ↓
Train / Evaluation Split
      ↓
1,800 Training | 200 Evaluation
      ↓
Format & Tokenize
      ↓
Base TinyLlama Evaluation
      ↓
4-bit NF4 Quantization
      ↓
LoRA Adapter
      ↓
QLoRA Fine-Tuning
      ↓
Held-out Evaluation
      ↓
Loss + Perplexity
      ↓
Base vs Fine-Tuned Comparison
      ↓
Merge & Save
```

## Key Features

* TinyLlama-1.1B-Chat fine-tuning
* QLoRA with 4-bit NF4 quantization
* BitsAndBytes double quantization
* PEFT LoRA adapters
* Rank-8 LoRA configuration
* Gradient accumulation
* Held-out evaluation dataset
* Evaluation loss
* Perplexity
* Base vs fine-tuned generation comparison
* LoRA adapter merging

## Dataset

The project uses the [Databricks Dolly 15K](https://huggingface.co/datasets/databricks/databricks-dolly-15k) instruction-following dataset.

For a lightweight Colab T4 experiment, 2,000 examples are selected:

* Training: 1,800 examples
* Evaluation: 200 examples

The evaluation examples are kept separate from training and are not used for gradient updates.

## Model

Base model:

`TinyLlama/TinyLlama-1.1B-Chat-v1.0`

The model is loaded using 4-bit quantization with:

* NF4 quantization
* Double quantization
* FP16 compute

The configuration is designed for running the experiment on a Google Colab T4 GPU.

## QLoRA Configuration

LoRA adapters are applied to:

* `q_proj`
* `v_proj`

Configuration:

| Parameter           |     Value |
| ------------------- | --------: |
| LoRA rank           |         8 |
| LoRA alpha          |        16 |
| LoRA dropout        |      0.05 |
| Quantization        | 4-bit NF4 |
| Double quantization |   Enabled |
| Compute dtype       |      FP16 |

The base model remains frozen while the LoRA parameters are trained.

## Training

The experiment uses:

* Batch size: 4
* Gradient accumulation steps: 4
* Epochs: 2
* Learning rate: 2e-4
* Maximum sequence length: 256
* FP16 training

Gradient accumulation is used to keep the effective batch size reasonable within T4 GPU memory constraints.

## Evaluation

Evaluation is performed on the 200 examples held out from training.

### Quantitative Evaluation

The notebook calculates:

* Evaluation loss
* Perplexity

Perplexity is calculated from the evaluation loss:

```text
perplexity = exp(evaluation_loss)
```

The values shown in the notebook are generated from the actual training run rather than hard-coded results.

### Qualitative Evaluation

The same unseen instructions are given to:

1. The original TinyLlama model
2. The QLoRA fine-tuned model

This allows a direct comparison of the generated responses.

The comparison focuses on:

* Instruction following
* Relevance
* Helpfulness
* Response quality

## Results

Run the notebook to generate the evaluation metrics and model comparison results.

The project intentionally avoids claiming improvement without evaluating the actual outputs.

For a small 2,000-example fine-tuning run, the results should be interpreted as a demonstration of the QLoRA workflow rather than a benchmark of TinyLlama.

## Repository Structure

```text
qlora-tinyllama-dolly/
│
├── qlora_tinyllama_dolly.ipynb
├── README.md
├── requirements.txt
└── .gitignore
```

Large model weights and datasets are not included in the repository.

## Installation

Clone the repository and install the dependencies:

```bash
pip install -r requirements.txt
```

The notebook is designed to run in a GPU environment such as Google Colab with a T4 GPU.

## Running the Notebook

1. Open `qlora_tinyllama_dolly.ipynb` in Google Colab.
2. Enable a GPU runtime.
3. Install the required dependencies.
4. Run the notebook from top to bottom.
5. Review the evaluation loss and perplexity.
6. Compare the base and fine-tuned model responses.
7. Optionally save the merged model.

## Limitations

This is a lightweight portfolio experiment rather than a production fine-tuning pipeline.

The experiment uses only 2,000 Dolly examples and two training epochs.

The training objective uses the complete formatted sequence rather than response-only loss.

TinyLlama's relatively small parameter count also limits the potential quality of the resulting model.

The evaluation should therefore be interpreted as an experiment demonstrating the QLoRA fine-tuning workflow rather than a general benchmark.

## Technologies

* Python
* PyTorch
* Hugging Face Transformers
* Hugging Face Datasets
* PEFT
* bitsandbytes
* Google Colab
* QLoRA
* LoRA
