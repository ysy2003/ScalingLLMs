# HPML Project: Design2Code at Scale: A Framework-Based Comparison of Lightweight Fine-Tuning and Zero-Shot VLMs

## Team Information

- **Team Name**: [Team Name]
- **Members**:
  - Chunyu Jin (cj2871)
  - Shuyang Yu (sy3309)
  - Jiayi Chen (jc6683)

---

## 1. Problem Statement

Describe the task being solved/researched

---

## 2. Model Description

Summarize the model architecture(s) used (e.g., ResNet-18, Transformer). Include:

- Framework (e.g., PyTorch, TensorFlow)
- Any custom layers or changes to standard models

---

## 3. Final Results Summary

Example Table: 

| Metric               | Value                           |
| -------------------- | ------------------------------- |
| Final Top-1 Accuracy | XX.XX%                          |
| Inference Latency    | XX.XX ms                        |
| Model Size           | XX MB                           |
| Peak Memory Use      | XX MB                           |
| Training Time/Epoch  | XX s                            |
| Device               | A100, Jetson Nano, M1 Pro, etc. |

---

## 4. Reproducibility Instructions

### A. Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

---

B. Wandb Dashboard

View training and evaluation metrics here: Wandb Dashboard Link
(Replace with actual link)

---

### C. Specify for Training or For Inference or if Both 

To train the model from scratch:

```bash
python train.py --config configs/default.yaml
```

---

### D. Evaluation

To evaluate the trained model:

```bash
python eval.py --weights checkpoints/best_model.pth
```

---

### E. Quickstart: Minimum Reproducible Result

To reproduce our minimum reported result (e.g., XX.XX% accuracy), run:

```bash
# Step 1: Set up environment
pip install -r requirements.txt

# Step 2: Download dataset
bash scripts/download_dataset.sh  # if applicable

# Step 3: Run training (or skip if checkpoint is provided)
python train.py --config configs/default.yaml

# Step 4: Evaluate
python eval.py --weights checkpoints/best_model.pth
```

---

## 5. Notes (up to you)

- All scripts are located in `scripts/`, `train.py`, `eval.py`, and `configs/`.
- Trained Model are saved in `models/`.
- Contact information




