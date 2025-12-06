# Model Integration

## Gemini
This module provides asynchronous processing capabilities for generating HTML/CSS code from design images using Google's Gemini API.

The `gemini.py` script processes PNG design images from Google Cloud Storage and uses the Gemini 2.5 Flash model to generate corresponding HTML and CSS code. The implementation uses asynchronous processing to handle multiple requests concurrently, significantly improving throughput.


## Prerequisites

### 1. Install Google Cloud CLI

If you don't have gcloud CLI installed:

- **Windows**: Download from [Google Cloud SDK](https://cloud.google.com/sdk/docs/install)
- **macOS**: `brew install google-cloud-sdk`
- **Linux**: Follow [official installation guide](https://cloud.google.com/sdk/docs/install)

### 2. Authenticate with Google Cloud

```bash
#gcloud auth login
gcloud auth application-default login
```

### 3. Set Project Configuration

```bash
gcloud auth application-default set-quota-project crack-battery-473522-r9

.env file
GOOGLE_GENAI_USE_VERTEXAI=True
GOOGLE_CLOUD_PROJECT=crack-battery-473522-r9
```

### 4. Verify Configuration

```bash
gcloud auth list
gcloud config list
```

### 5. Install Python Dependencies

```bash
pip install -r requirements.txt
```

Required packages:
- `google-genai` - Google Gemini API client
- `pandas` - Data manipulation and Excel export
- `python-dotenv` - Environment variable management
- `openpyxl` - Excel file support (for pandas)

## Dataset Setup

### Accessing the Dataset

The dataset is stored in Google Cloud Storage. To view available files:

```bash
gcloud storage ls --recursive gs://dataset_design2code
```

You should see output like:
```
gs://dataset_design2code/:
gs://dataset_design2code/.gitattributes
gs://dataset_design2code/10018.html
gs://dataset_design2code/10018.png
gs://dataset_design2code/1002.html
gs://dataset_design2code/1002.png
...
```

### Dataset File Format

The `dataURI.txt` file contains a list of Google Cloud Storage URIs for both PNG images and their corresponding HTML files. Each design image has:
- A PNG file: `gs://dataset_design2code/{number}.png`
- An HTML file: `gs://dataset_design2code/{number}.html`

## Usage

### Basic Usage

Process a limited number of images:

```python
import asyncio
from models.gemini import process_images_from_file

# Process 10 images with default settings (max 10 concurrent requests)
df = asyncio.run(process_images_from_file(n_images=10))
df.to_excel("models/gemini_results.xlsx", index=False)
```

### Advanced Usage

Customize processing parameters:

```python
import asyncio
from models.gemini import process_images_from_file

# Process 50 images with 5 concurrent requests and custom prompt
df = asyncio.run(
    process_images_from_file(
        n_images=50,
        max_concurrent=5,
        prompt="models/prompt.txt"
    )
)
df.to_excel("models/gemini_results.xlsx", index=False)
```

### Command Line Execution

Run the script directly:

```bash
python models/gemini.py
```

The script will:
1. Process images according to the configuration in `__main__`
2. Save results to `models/gemini_results.xlsx`
3. Display summary statistics

### Post-Processing: HTML File Cleaning

After generating predictions, you may need to clean the HTML files to remove Markdown code block markers that might have been included in the generated output. Use `html_cleaner.py` for this purpose:

```bash
python gemini/html_cleaner.py
```

**Purpose**: The `html_cleaner.py` script removes Markdown code block markers from the beginning and end of HTML files in the prediction folders. For example, it removes opening markers like "```html" and closing markers like "```". This is necessary because some models may include these markers in their output, which would cause rendering errors during evaluation.

**When to use**: Run this script after generating predictions but before running evaluation scripts (e.g., `metrics/correctness.py`).

**Configuration**: Edit the `PREDICTION_FOLDERS` list in `html_cleaner.py` to specify which prediction folders should be cleaned:

```python
PREDICTION_FOLDERS: List[Path] = [
    BASE_DIR / 'gemini_predictions1',
    BASE_DIR / 'gemini_predictions2',  # Add more folders as needed
]
```

## Configuration

### Environment Variables

Create a `.env` file in the project root with your Google Cloud credentials:

```
GOOGLE_APPLICATION_CREDENTIALS=path/to/credentials.json
```

Or ensure you're authenticated via `gcloud auth login`.

### Prompt Customization

Edit `models/prompt.txt` to customize the prompt sent to the Gemini API. The default prompt is:

```
Strictly follow the design in the image. Generate the semantic HTML and responsive CSS. The code must be production-ready and accurately recreate all visual elements and layouts. Output only the code.
```

### Processing Parameters

- `n_images`: Number of images to process (None for all images)
- `max_concurrent`: Maximum number of concurrent API requests (default: 10)
- `prompt`: Path to prompt text file (default: "models/prompt.txt")
- `file_path`: Path to dataURI.txt file (default: "models/dataURI.txt")

## Output Format

The results DataFrame contains the following columns:

- `png_uri`: Google Cloud Storage URI of the PNG image
- `html_uri`: Google Cloud Storage URI of the reference HTML file
- `response_text`: Generated HTML/CSS code from Gemini API
- `number`: Image identifier number
- `prompt_token_count`: Number of tokens in the prompt
- `candidates_token_count`: Number of tokens in the generated response
- `thoughts_token_count`: Number of tokens used for reasoning (if applicable)
- `total_token_count`: Total tokens used
- `latency`: Processing time in seconds
- `error`: Error message (if API call failed)



## Troubleshooting

### Authentication Issues

```bash
# Re-authenticate
gcloud auth login
gcloud auth application-default login
```

### Permission Errors

Ensure your Google Cloud account has access to:
- The project: `crack-battery-473522-r9`
- The storage bucket: `gs://dataset_design2code`
- Gemini API access


## Reference

For generating the dataURI.txt file (reference only, not needed if file exists):

```bash
gcloud storage ls --recursive gs://dataset_design2code/ | findstr /R "\.png$ \.html$" | findstr /V /C:"Design2Code/" | findstr /V /C:".cache/" > dataURI.txt
```
# Results
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
## 2.2 IOU
## Summary Statistics
   Total Test Cases (from Design2Code): 484
   Successfully Processed:              468
   Skipped (file not found):            16
   Errors:                              0

## 🏆 IOU Score Metrics (for valid scores only)
   Average IOU Score:                  0.1567
   Minimum IOU Score:                  0.0000
   Maximum IOU Score:                  0.4717

## Score Distribution
   High (≥0.8):                        0 (0.0%)
   Medium (0.5-0.8):                   0 (0.0%)
   Low (<0.5):                         468 (100.0%)
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
# Structural Alignment Score Summary
## Summary Statistics
   Total Test Cases (from Design2Code): 484
   Successfully Processed:              468
   Skipped (file not found):            16
   Errors:                              0

## 🏆 Tree Edit Similarity Metrics (1.0 = identical)
   Average:                            0.2099
   Minimum:                            0.0000
   Maximum:                            0.8235

## 🏆 Semantic HTML Ratio Metrics (Higher is better)
   Average:                            0.2325
   Minimum:                            0.0000
   Maximum:                            1.0000

## 🏆 Accessibility Score Metrics (Higher is better)
   Average:                            0.0017
   Minimum:                            0.0000
   Maximum:                            0.5000