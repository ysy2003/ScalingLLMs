# [Scaling LLM Project] Design2Code at Scale: A Framework-Based Comparison of Lightweight Fine-Tuning and Zero-Shot VLMs

## Team Information

- **Team Name**: Design2Code at Scale
- **Members**:
  - Chunyu Jin (cj2871)
  - Shuyang Yu (sy3309)
  - Jiayi Chen (jc6683)

---

## 1. Problem Statement

How should practitioners choose between general-purpose VLM and specialized fine-tuned models for automatic UI-to-code generation?

---

## 2. Model Description

We evaluate four Vision-Language Models (VLMs) on design-to-code generation:

### Foundation Models (Zero-Shot)
- **Gemini-2.5-Flash**: Google's multimodal model accessed via API, capable of processing design images and generating HTML/CSS code without task-specific training.
- **Qwen3-VL-8B-Thinking**: Open-source 8B parameter VLM from Alibaba with enhanced reasoning capabilities, run locally on GPU.

### Fine-tuned Models
- **Design2Code-18B-v0**: An 18B parameter model fine-tuned specifically on design-to-code tasks, based on CogVLM architecture.
- **VLM-WebSight**: A VLM fine-tuned on the WebSight dataset for HTML generation from webpage screenshots.

### Framework and Implementation
- **Framework**: PyTorch 2.8.0, HuggingFace Transformers 4.57.1
- **Hardware**: NVIDIA A100 GPU (80GB VRAM)
- **Dataset**: [SALT-NLP/Design2Code-hf](https://huggingface.co/datasets/SALT-NLP/Design2Code-hf) (484 samples)

---

## 3. Final Results Summary

### Evaluation Results

![Results Table](results.png)

| Model | CLIP | IoU | Render Success | Error Rate | Tree Edit Sim | Semantic HTML | Degradation | Consistency | Latency | VRAM/Cost |
|-------|------|-----|----------------|------------|---------------|---------------|-------------|-------------|---------|-----------|
| **Gemini-2.5-Flash** | 0.82 | 0.16 | 77.89% | 3.51% | 0.21 | 23.25% | 0.00% | 0.66 | 34.22s | $0.0020/1k |
| **Qwen3-VL-8B-Thinking** | 0.68 | 0.13 | 96.90% | 1.03% | 0.18 | 27.13% | 21.05% | 0.66 | 153.95s | 17.70 GB |
| **Design2Code-18B-v0** | 0.77 | 0.12 | 100.00% | 24.17% | 0.16 | 66.56% | 15.56% | 0.72 | 208.26s | 39.44 GB |
| **VLM-WebSight** | 0.71 | 0.10 | 98.97% | 51.45% | 0.09 | 11.90% | 22.50% | 0.52 | 366.73s | 16.22 GB |

### Key Metrics
- **Visual Fidelity**: CLIP similarity and IoU between rendered output and reference
- **Code Correctness**: Render success rate and syntax error rate
- **Structural Alignment**: DOM tree edit similarity and semantic HTML tag usage
- **Robustness**: Performance degradation under image perturbations (50 samples)
- **Computational Efficiency**: Inference latency and memory/cost requirements

---

## 4. Reproducibility Instructions

### A. Requirements

Install dependencies:

```bash
pip install -r requirements.txt

# Install Playwright browsers (required for rendering)
playwright install chromium
```

---

### B. Project Structure

```
Design2code/
├── metrics/              # Evaluation metrics implementation
│   ├── CLIP.py          # Visual fidelity (CLIP similarity)
│   ├── IOU.py           # Visual fidelity (IoU)
│   ├── structural_alignment.py  # Tree edit distance & semantic HTML
│   ├── correctness.py   # Render success & error detection
│   └── robustness.py    # Degradation rate calculation
├── Qwen/                # Qwen3-VL inference scripts
├── design2code_18b/     # Design2Code-18B model scripts
├── gemini/              # Gemini API integration
├── robustness/          # Robustness testing pipeline
├── results_Qwen/        # Qwen model predictions
├── results_WebSight/    # WebSight model predictions
├── results_Design2Code18B/  # Design2Code predictions
└── eval/                # Evaluation notebooks
```

---

### C. Running Inference

**Qwen3-VL-8B:**
```bash
cd Qwen/
python test_qwen_metrics.py
```

**Design2Code-18B-v0:**
```bash
cd design2code_18b/
bash run_design2code18b_full.sh
```

**Gemini-2.5-Flash:**
```bash
cd gemini/
python gemini.py
```

---

### D. Evaluation

Run evaluation metrics on model predictions:

```bash
# Visual Fidelity (CLIP Score)
python metrics/CLIP.py

# Visual Fidelity (IoU)
python metrics/IOU.py

# Structural Alignment
python metrics/structural_alignment.py

# Code Correctness
python metrics/correctness.py
```

---

### E. Robustness Testing

To evaluate model robustness under image perturbations:

```bash
cd robustness/

# Test Qwen model
python test_robustness.py --model qwen --samples 50

# Test WebSight model
python test_robustness.py --model websight --samples 50
```

---

## 5. Notes

- All evaluation results are saved in `results_*/` directories
- The dataset is automatically downloaded from HuggingFace
- For Gemini API usage, ensure Google Cloud credentials are configured
- Visualization notebook: `visualization.ipynb`




