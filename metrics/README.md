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

   Total Predictions Evaluated:   484

### 🏆 Metrics
   Pass@1 (Avg Accuracy):   96.69%

### 📊 Error Statistics
   ```
   Error Count:
      Total:                 17
      Average:               0.04
      Max:                    1
      Files with errors:     17

   Critical Error Count:
      Total:                 16
      Average:               0.03
      Max:                    1
      Files with critical errors: 16
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