# Metrics
## Correctness
Code Correctness:
Render Success Rate: Percentage of generated code that renders without critical errors.

Pass@k: Code validity measured by repeated sampling (for models with stochastic decoding).

DOM/Console Error Count: Number of errors logged by the browser's developer tools upon rendering.
### environment setup and running
```
pip install playwright
playwright install
```
run correctness.py

### performance

🚀 Launching headless browser (Playwright)...

📂 Processing Batch 1/3: E:\code\Columbia\ScallingLLMsProject\ScalingLLMs\gemini\results\gemini_predictions1
   📊 Found: 483, Missing: 1

📂 Processing Batch 2/3: E:\code\Columbia\ScallingLLMsProject\ScalingLLMs\gemini\results\gemini_predictions2
   📊 Found: 443, Missing: 41

📂 Processing Batch 3/3: E:\code\Columbia\ScallingLLMsProject\ScalingLLMs\gemini\results\gemini_predictions3
   📊 Found: 435, Missing: 49

✅ Browser closed. Calculation complete.

### 📊 Gemini Design2Code Pass@3 Report

#### Summary Statistics
   Total Unique Test Cases (from dataURI.txt): 484
   Total Predictions Evaluated:   1452

#### 🏆 Metrics
   Pass@1 (Avg Accuracy):   93.73%
   Pass@3 (Best of 3):      99.79%

   (Pass@3 means 483 out of 484 UI designs rendered correctly at least once across all 3 batches.)

✅ Detailed report saved to: metrics/evaluation/pass_k_metrics_report_gemini.xlsx