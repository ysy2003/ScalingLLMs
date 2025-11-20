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


## File Structure

```
models/
├── README.md           # This file
├── gemini.py           # Main processing script
├── prompt.txt          # Prompt template for API
├── dataURI.txt         # List of dataset URIs
└── gemini_results.xlsx # Output results (generated)
```

## Reference

For generating the dataURI.txt file (reference only, not needed if file exists):

```bash
gcloud storage ls --recursive gs://dataset_design2code/ | findstr /R "\.png$ \.html$" | findstr /V /C:"Design2Code/" | findstr /V /C:".cache/" > dataURI.txt
```


