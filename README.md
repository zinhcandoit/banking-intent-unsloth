# Banking Intent Classification — Qwen3-4B + Unsloth

Fine-tune **Qwen3-4B** (4-bit quantized via Unsloth) with LoRA adapters on the **Banking77** dataset to classify 77 banking customer intent categories.

Supports three inference modes: **zero-shot** and **fine-tuned**.

## Video Demonstration

[Video Demo](https://drive.google.com/file/d/18vQH2IbVTlGWPi1hxII4-61F-qfSh0YQ/view?usp=sharing)

---

## Project Structure

```
banking-intent-unsloth/
├── scripts/
│   ├── train.py              # Fine-tune Qwen3-4B with LoRA via Unsloth
│   ├── inference.py          # IntentClassification class
│   ├── evaluate.py           # Evaluate accuracy on test set
│   └── preprocess_data.py    # Download Banking77 from HuggingFace
├── configs/
│   ├── train.yaml            # Training hyperparameters
│   └── inference.yaml        # Inference configuration (gitignored — contains secrets)
├── sample_data/
│   ├── train.csv             # Training set (5,000 samples)
│   └── test.csv              # Test set (~3,000 samples)
├── outputs/
│   └── checkpoint/           # Saved LoRA adapter after training
├── train_local.ipynb        # End-to-end Python notebook with resume support
├── train.sh                  # preprocess + train
├── inference.sh              # run inference (all 2 modes)
└── requirements.txt
```

---

## Setup

Requires a GPU environment (Kaggle T4/P100 or Google Colab recommended).

```bash
pip install -r requirements.txt
```

---

## Pipeline

### 1. Data Preparation

```bash
cd scripts
python preprocess_data.py
```

Downloads and Extracts training subset of Banking77 from HuggingFace (`PolyAI/banking77`) via direct parquet URLs and saves:
- `sample_data/train.csv` — 5,000 rows (`text`, `label`, `intent_name`) with the same distribution from origin.
- `sample_data/test.csv` — 3,084 rows

---

### 2. Fine-tuning

```bash
sh train.sh
# or
cd scripts && python train.py
```

The LoRA adapter is saved to `outputs/checkpoint/`. Training automatically resumes from the latest checkpoint if one exists.

#### Hyperparameters (`configs/train.yaml`)

| Parameter | Value |
|-----------|-------|
| Base model | `unsloth/Qwen3-4B-unsloth-bnb-4bit` |
| Quantization | 4-bit (QLoRA) |
| Max sequence length | 2048 |
| LoRA rank (r) | 16 |
| LoRA alpha | 32 |
| LoRA dropout | 0.05 |
| LoRA target modules | q/k/v/o/gate/up/down proj |
| Epochs | 3 |
| Batch size | 2 |
| Gradient accumulation steps | 4 |
| Effective batch size | 8 |
| Learning rate | 2e-4 |
| LR scheduler | cosine |
| Optimizer | `adamw_8bit` |
| Weight decay | 0.01 |
| Checkpoint save every | 100 steps |

---

### 3. Inference

The `IntentClassification` class in `scripts/inference.py` supports 2 modes:

| Mode | Model | Description |
|------|-------|-------------|
| `finetuned` (default) | LoRA adapter | Fine-tuned on Banking77 |
| `zero_shot` | Base Qwen3-4B | No examples in prompt |

**Usage example:**
- Download finetuned model from HuggingFace - TQZinh/banking-intent-unsloth and change the `model_path` in `configs/inference.yaml`
- Run the script likes:
1. 
```python
from scripts.inference import IntentClassification

# model_path = path to inference.yaml config file
clf = IntentClassification("configs/inference.yaml")
label = clf("I lost my credit card, how do I order a replacement?")
print(label)  # e.g. "lost_or_stolen_card"
```
2. 
```bash
sh inference.sh
# or
cd scripts
python inference.py
```

Expected output:

```
[ZERO_SHOT]
  input      : 'I lost my credit card yesterday, how do I order a new one?'
  raw_output : 'lost_or_stolen_card'
  label      : 'lost_or_stolen_card'

[FINETUNED]
  ...
```

---

### 4. Evaluation

```bash
cd scripts

# Fine-tuned model (default)
python evaluate.py

# Base model — zero-shot
python evaluate.py --mode zero_shot

# All 2 modes with comparison summary
python evaluate.py --mode all
```

---

### 5. LangSmith Tracing (optional)

Set `LANGSMITH_API_KEY` as an environment variable or add it to `configs/inference.yaml` under `langsmith_api_key`. Every `__call__` invocation will be traced to the project specified by `langsmith_project`.
