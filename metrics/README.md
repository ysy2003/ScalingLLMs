# Metrics
# 1. Correctness
Code Correctness:
Render Success Rate: Percentage of generated code that renders without critical errors.

Pass@k: Code validity measured by repeated sampling (for models with stochastic decoding).

DOM/Console Error Count: Number of errors logged by the browser's developer tools upon rendering.
## environment setup and running
```
pip install playwright
playwright install
```
run correctness.py

## 📊 Gemini Design2Code Pass@1 Report


### Summary Statistics
   Total Unique Test Cases (from dataURI.txt): 484

   Total Predictions Evaluated:   Found: 468, Missing: 16

### 🏆 Metrics
   Pass@1 (Avg Accuracy):   77.89%

### 📊 Error Statistics
   ```
   Error Count:
      Error Count:
      Total:                 18
      Average:               0.04
      Max:                    2
      Files with errors:     17

   Critical Error Count:
      Total:                 107
      Max:                    1
      Files with critical errors: 107
   ```

✅ Detailed report saved to: [metrics/evaluation/gemini_correctness_report.xlsx](metrics/evaluation/gemini_correctness_report.xlsx)

# 2. Visual Fidelity
## 2.1 CLIP
### Gemini 📊 CLIP Score Summary

#### Summary Statistics
   ```
   Total Test Cases (from Design2Code): 484

   Successfully Processed:              468

   Skipped (file not found):            16

   Errors:                              0

   Missing HTML files:                  16
   ```

#### 🏆 CLIP Score Metrics (for valid scores only)
   ```
   Average CLIP Score:                  0.8173

   Minimum CLIP Score:                  0.3564

   Maximum CLIP Score:                  0.9753
   ```

#### Score Distribution
   ```
   High (≥0.8):                        302 (64.5%)

   Medium (0.5-0.8):                    155 (33.1%)

   Low (<0.5):                          11 (2.4%)
   ```
✅ Detailed report saved to: [metrics\evaluation\gemini_clip_scores.xlsx](metrics\evaluation\gemini_clip_scores.xlsx)
# 3. Efficiency
## Dataset Summary:
  Total requests: 468

  Valid requests: 468

## Token Usage:
```
  Total input tokens: 850,428

  Total output tokens: 3,148,123.0

  Total tokens: 3,998,551.0

  Average input tokens per request: 1817.15

  Average output tokens per request: 6726.76
```
## Latency:
```
  Total latency: 16017.70 seconds
  Average latency: 34.2259 seconds
  Min latency: 10.1520 seconds
  Max latency: 91.5540 seconds
  Median latency: 33.7336 seconds
```
## Cost Breakdown:
```
  Input cost: $0.255128 ($0.3/1M tokens)
  Output cost: $7.870308 ($2.5/1M tokens)
  Total cost: $8.125436
  Cost per 1k tokens: $0.002032
  Cost per request: $0.017362
```
## EfficiencyScores Object:
```
  tokens_per_second: 196.540204561509
  latency_seconds: 34.2259
  cost_per_1k_tokens: $0.002032
```