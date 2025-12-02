# ScalingLLMs

*Dataset*
- Design2Code: google cloud + huggingface
- Web2Code: only in huggingface
- Pix2Code: google cloud + huggingface
- Chart2Code: only in google cloud
- Plotbench: google cloud + huggingface
- WebSight: only in huggingface

# Awesome Design2Code

## Table of Contents

- [1. Benchmarks and Datasets for UI-to-Code](#1-benchmarks-and-datasets-for-ui-to-code)  
- [2. Specialized UI-to-Code Models](#2-specialized-ui-to-code-models)  
- [3. UI Design Quality and Evaluation Metrics](#3-ui-design-quality-and-evaluation-metrics)  
- [4. Summary and Gap Analysis](#4-summary-and-gap-analysis)  

---

## 1. Benchmarks and Datasets for UI-to-Code

### 1.1 Static Webpage Design → Code (Screenshot / UI → HTML/CSS/Frontend Code)

| Name | Venue / Year | Type | Link |
| --- | --- | --- | --- |
| WebSight: Unlocking the Conversion of Web Screenshots into HTML Code with the WebSight Dataset | arXiv 2024 | Synthetic screenshot → HTML dataset | https://arxiv.org/abs/2403.09029 |
| WebCode2M: A Real-World Dataset for Code Generation from Webpage Designs | WWW / arXiv 2024–2025 | Large-scale real-world webpage design → code dataset | https://arxiv.org/abs/2404.06369 |
| Design2Code: Benchmarking Multimodal Code Generation for Automated Front-End Engineering | NAACL 2025 | Benchmark for screenshot → HTML/CSS with metrics | https://aclanthology.org/2025.naacl-long.199/ |
| Web2Code: A Large-scale Webpage-to-Code Dataset and Evaluation Framework for Multimodal LLMs | arXiv 2024 | Webpage screenshot + instruction → HTML + QA benchmark | https://arxiv.org/abs/2406.20098 |
| WebUIBench: A Comprehensive Benchmark for Evaluating Multimodal Large Language Models in WebUI-to-Code | Findings of ACL 2025 | Benchmark for WebUI (image + text) → code | https://arxiv.org/abs/2506.07818 |
| DesignBench: A Comprehensive Benchmark for MLLM-based Front-end Code Generation | arXiv 2025 | Multi-framework (React/Vue/HTML) design-to-code benchmark | https://arxiv.org/abs/2506.06251 |

### 1.2 Interactive / Multi-step Web UI Benchmarks

| Name | Venue / Year | Type | Link |
| --- | --- | --- | --- |
| Interaction2Code: How Far Are We From Automatic Interactive Webpage Generation? | arXiv 2024 | Benchmark for interactive behaviors (hover, modal, animation) → code | https://arxiv.org/abs/2411.03292 |

---

## 2. Specialized UI-to-Code Models

| Model / Paper | Venue / Year | Input → Output | Notes | Link |
| --- | --- | --- | --- | --- |
| MLLM-Based UI2Code Automation Guided by UI Layout Information (often called LayoutCoder) | ISSTA 2025 | UI screenshot + layout tree → code | Layout-guided UI2Code with explicit layout representation | https://arxiv.org/abs/2506.10376 |
| UICopilot: Automating UI Synthesis via Hierarchical Code Generation from Webpage Designs | WWW / arXiv 2025 | Webpage design → hierarchical HTML/CSS | Two-stage (structure-first) UI2Code system | https://arxiv.org/abs/2505.09904 |
| LaTCoder: Converting Webpage Design to Code with Layout-as-Thought | KDD / arXiv 2025 | Webpage design → code via block-wise reasoning | Layout-as-Thought (LaT), block-by-block generation | https://arxiv.org/abs/2508.03560 |
| ViT-DtC: Vision Transformer-based Design-to-Code Framework for Generated UI Designs and Hand-drawn Sketches | Neural Computing & Applications 2025 | Generated UI / sketch → multi-platform code | ViT-based design-to-code for web + mobile | *(journal DOI / link as available)* |
| UIGEN-T2: UI Generation Model Fine-tuned from Qwen2.5-Coder-7B-Instruct | Model card 2025 | Text / UI spec → front-end code | HF model for UI/UX generation tasks | https://huggingface.co/Tesslate/UIGEN-T2-7B |

---

## 3. UI Design Quality and Evaluation Metrics

### 3.1 Metric Suites Embedded in Design-to-Code Benchmarks


| Benchmark / Paper | Venue / Year | Metric Focus | Link |
| --- | --- | --- | --- |
| Design2Code | NAACL 2025 | Element recall, layout similarity, DOM tree similarity, human evaluation | https://aclanthology.org/2025.naacl-long.199/ |
| Web2Code | arXiv 2024 | Webpage QA, element localization, HTML generation metrics | https://arxiv.org/abs/2406.20098 |
| WebCode2M | WWW / arXiv 2024–2025 | Long-sequence HTML quality, robustness on real-world pages | https://arxiv.org/abs/2404.06369 |
| WebSight | arXiv 2024 | Screenshot→HTML accuracy on large synthetic dataset | https://arxiv.org/abs/2403.09029 |
| WebUIBench | Findings of ACL 2025 | WebUI-to-code metrics, multi-model comparison | https://arxiv.org/abs/2506.07818 |
| Interaction2Code | arXiv 2024 | Interaction success rate, behavioral correctness metrics | https://arxiv.org/abs/2411.03292 |
| DesignBench | arXiv 2025 | Generate/edit/repair metrics, code-level error taxonomy | https://arxiv.org/abs/2506.06251 |

### 3.2 Learned UI Quality Scorers / UI Aesthetic Evaluation

| Paper / Model | Venue / Year | Role | Link |
| --- | --- | --- | --- |
| UIClip: A Data-driven Model for Assessing User Interface Design | UIST 2024 | CLIP-style learned scorer for UI quality / aesthetics | https://arxiv.org/abs/2404.12500 |

---

## 4. Summary and Gap Analysis

- Recent work has produced **rich datasets and benchmarks** for static screenshot→code, real-world webpages, WebUI, and interactive behaviors (Section 1).  
- A series of **specialized UI2Code systems** (LayoutCoder, UICopilot, LaTCoder, ViT-DtC, etc.) demonstrate that structure-aware and reasoning-style generation can significantly improve design-to-code quality (Section 2).  
- Multiple benchmarks come with their own **metric suites and error taxonomies**, and there are emerging **learned UI quality scorers** such as UIClip (Section 3).  

However, these resources are **fragmented across different codebases and evaluation scripts**. Our project aims to:

1. **Unify** key metrics from existing benchmarks into a single, reusable evaluation framework.  
2. Support **composite workflows** (generation, QA, edit, repair, interaction) rather than only single-step screenshot→code evaluation.  
3. Provide **fine-grained, UI-aware error analysis** that aligns with how front-end engineers reason about layout, components, content, and interaction issues.
