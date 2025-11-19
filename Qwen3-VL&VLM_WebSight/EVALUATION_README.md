# Model Evaluation Guide

## Overview
Evaluates Qwen3-VL-2B and VLM_WebSight_finetuned models on the SALT-NLP/Design2Code-hf dataset using all applicable metrics from the `metrics/` package.

## Prerequisites

### 1. Install Dependencies
```bash
pip install -r requirements.txt
playwright install  # Install browser binaries
```

### 2. Download Models
Ensure the VLM_WebSight_finetuned model is available at:
```
../models--HuggingFaceM4--VLM_WebSight_finetuned/snapshots/a5c2b06bfee0bd713cf2a6b3e4d46f94dd8fe839/
```

## Usage

### Run Evaluation
```bash
python evaluate_models.py
```

This will:
- Load both models
- Generate HTML/CSS for all samples in Design2Code dataset
- Compute all applicable metrics from `../metrics/` package:
  - Efficiency metrics (latency, throughput, VRAM)
  - Structural alignment (semantic HTML, accessibility)
  - Correctness (render success, error count)
- Save results to `evaluation_results/evaluation_results.xlsx`
- Save generated HTML files to `evaluation_results/{model_name}/`

## Output Structure

```
evaluation_results/
├── evaluation_results.xlsx          # Combined metrics for both models
├── Qwen3-VL-2B/
│   ├── sample_0.html
│   ├── sample_1.html
│   └── ...
└── VLM_WebSight/
    ├── sample_0.html
    ├── sample_1.html
    └── ...
```

## Metrics Evaluated

All metrics are from the `../metrics/` package:

### 1. Efficiency Metrics (`metrics.efficiency`)
- **Latency**: Time to generate code (seconds)
- **Tokens per Second**: Generation throughput
- **Peak VRAM**: GPU memory usage (GB)

### 2. Structural Alignment (`metrics.structural_alignment`)
- **Semantic HTML Ratio**: Proportion of semantic tags vs div tags
- **Accessibility Score**: Coverage of alt attributes and ARIA labels

### 3. Correctness (`metrics.correctness` approach)
- **Render Success**: Whether HTML renders without critical errors
- **Error Count**: Number of console/DOM errors during render

**Note**: Tree edit similarity is not included as it requires implementation. Robustness metrics require perturbed images. Visual fidelity metrics are not yet implemented.

## Notes

- Models are loaded in FP16 to reduce memory usage
- Qwen3-VL generates with temperature=0.7, max_new_tokens=2048
- VLM_WebSight uses max_length=2048
- All images are converted to RGB before processing
- Correctness is checked per-sample using Playwright
- The script processes the full dataset sequentially

## Memory Requirements

- GPU with at least 16GB VRAM recommended
- Models are loaded one at a time to conserve memory
- Torch cache is cleared between model evaluations

## Troubleshooting

**Out of Memory Error:**
- Reduce `max_new_tokens` in generation parameters
- Process dataset in smaller batches

**Playwright Errors:**
- Ensure browser binaries are installed: `playwright install`
- Check that HTML files are being saved correctly

**Import Errors:**
- Ensure running from this directory
- Script adds parent directory to path for metrics imports
