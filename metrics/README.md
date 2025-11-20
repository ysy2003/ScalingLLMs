# Metrics
## Correctness
Code Correctness:
Render Success Rate: Percentage of generated code that renders without critical errors.

TODO: Pass@k: Code validity measured by repeated sampling (for models with stochastic decoding).

DOM/Console Error Count: Number of errors logged by the browser's developer tools upon rendering.
### environment setup and running
```
pip install playwright
playwright install
```
run correctness.py

### performance

--- 📊 Aggregate Metrics Report for Gemini---

#### 1. Render Success Rate
   483 / 483 files rendered successfully (<body> not empty)
   Rate: 100.00%

#### 2. DOM/Console Error Count
   Total Errors Found: 1
   Average Errors per File: 0.00

#### 3. Critical Error Count
   Total Critical Errors Found: 0
   Average Critical Errors per File: 0.00
#### 4. pass@k