from google import genai
from google.genai.types import HttpOptions, Part
import pandas as pd
import os
import time
import asyncio
from typing import Optional
from google.genai import types
from dotenv import load_dotenv
load_dotenv()
MAX_NEW_TOKENS = 9000
PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT")
client = genai.Client(project=PROJECT_ID,
    http_options=HttpOptions(api_version="v1"))
predictions_dir = "gemini/results/gemini_predictions3" # results directory folder for html files
if not os.path.exists(predictions_dir):
    os.makedirs(predictions_dir)

async def generate_content_from_image(png_uri, prompt="gemini/prompt.txt", model="gemini-2.5-flash"):
    """
    Generate content using Gemini API (async)
    
    Args:
        png_uri: URI path of the PNG image
        prompt: Prompt text file path
        model: Model name to use, defaults to "gemini-2.5-flash"
    
    Returns:
        Response object, or None if an error occurs
    """
    prompt_text = open(prompt, 'r').read()
    generate_content_config = types.GenerateContentConfig(
    temperature = 0.7,
    max_output_tokens = MAX_NEW_TOKENS,)
    try:
        # Run the synchronous API call in a thread pool to make it non-blocking
        response = await asyncio.to_thread(
            client.models.generate_content,
            model=model,
            contents=[
                prompt_text,
                Part.from_uri(
                    file_uri=png_uri,
                    mime_type="image/png",
                ),
            ],
            config=generate_content_config
        )
        return response
    except Exception as e:
        print(f"Error calling API for {png_uri}: {str(e)}")
        return None

async def process_single_image(number: str, png_uri: str, html_uri: Optional[str], prompt: str = "gemini/prompt.txt", logs_file: str = "gemini/gemini_results.txt"):
    """
    Process a single image asynchronously
    
    Args:
        number: Image number identifier
        png_uri: PNG image URI
        html_uri: Corresponding HTML URI
        prompt: Prompt text file path
    
    Returns:
        Dictionary with result data
    """
    print(f"Processing {png_uri}...")
    start_time = time.time()
    
    # Call Gemini API asynchronously
    response = await generate_content_from_image(png_uri, prompt=prompt)
    latency = time.time() - start_time
    
    if response is not None:
        # Store result
        if response.text is not None:
            with open(os.path.join(predictions_dir, png_uri.split("/")[-1].replace(".png", ".html")), "w", encoding='utf-8') as f:
                f.write(response.text)
           
        result= {
            'png_uri': png_uri,
            'html_uri': html_uri,
            'response_text': response.text if response.text else None,
            'number': number,
            'prompt_token_count': response.usage_metadata.prompt_token_count if response.usage_metadata and response.usage_metadata.prompt_token_count else None,
            'candidates_token_count': response.usage_metadata.candidates_token_count if response.usage_metadata and response.usage_metadata.candidates_token_count else None,
            'thoughts_token_count': response.usage_metadata.thoughts_token_count if response.usage_metadata and response.usage_metadata.thoughts_token_count else None,
            'total_token_count': response.usage_metadata.total_token_count if response.usage_metadata and response.usage_metadata.total_token_count else None,
            'latency': latency,
            'raw_response': response
        }
        # write result to txt file
        write_result={
            'png_uri': png_uri,
            'html_uri': html_uri,
            'number': number,
            'prompt_token_count': response.usage_metadata.prompt_token_count if response.usage_metadata and response.usage_metadata.prompt_token_count else None,
            'candidates_token_count': response.usage_metadata.candidates_token_count if response.usage_metadata and response.usage_metadata.candidates_token_count else None,
            'thoughts_token_count': response.usage_metadata.thoughts_token_count if response.usage_metadata and response.usage_metadata.thoughts_token_count else None,
            'total_token_count': response.usage_metadata.total_token_count if response.usage_metadata and response.usage_metadata.total_token_count else None,
            'latency': latency,
            'raw_response': response
        }

        with open(logs_file, 'a', encoding='utf-8') as f:
            f.write(f"{str(write_result)}\n")
        return result
    else:
        # API call failed
        return {
            'png_uri': png_uri,
            'html_uri': html_uri,
            'response_text': None,
            'number': number,
            'prompt_token_count': None,
            'candidates_token_count': None,
            'thoughts_token_count': None,
            'total_token_count': None,
            'latency': None,
            'raw_response': response
        }

async def process_images_from_file(file_path="gemini/dataURI.txt", n_images=None, max_concurrent=10, prompt="gemini/prompt.txt", logs_file: str = "gemini/gemini_results.txt"):
    """
    Read dataURI.txt file, process all PNG images asynchronously, and return a DataFrame containing response.text and corresponding HTML URIs
    
    Args:
        file_path: Path to the dataURI.txt file
        n_images: Number of images to process (None for all)
        max_concurrent: Maximum number of concurrent requests
        prompt: Prompt text file path
    
    Returns:
        DataFrame with results
    """
    
    # Read file and parse URIs
    png_uris = {}
    html_uris = {}
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            if line.endswith('.png'):
                # Extract number (extract filename from URI, remove .png)
                file_name = os.path.basename(line)
                number = file_name.replace('.png', '')
                png_uris[number] = line
            elif line.endswith('.html'):
                # Extract number
                file_name = os.path.basename(line)
                number = file_name.replace('.html', '')
                html_uris[number] = line
    
    # Limit number of images if specified
    if n_images is not None:

        png_uris = {k: v for k, v in list(png_uris.items())[:n_images]}
    else:
        png_uris = {k: v for k, v in list(png_uris.items())}
    # Create semaphore to limit concurrent requests
    semaphore = asyncio.Semaphore(max_concurrent)
    completed_count = 0
    total_count = len(png_uris)
    completed_lock = asyncio.Lock()
    
    async def process_with_semaphore(number: str, png_uri: str, html_uri: Optional[str]):
        """Process image with semaphore to limit concurrency"""
        nonlocal completed_count
        async with semaphore:
            result = await process_single_image(number, png_uri, html_uri, prompt, logs_file)
            async with completed_lock:
                completed_count += 1
                remaining = total_count - completed_count
                print(f"Completed {number}: {completed_count}/{total_count} (remaining: {remaining})")
            return result
    
    # Create tasks for all images
    tasks = [
        process_with_semaphore(number, png_uri, html_uris.get(number, None))
        for number, png_uri in png_uris.items()
    ]
    
    # Execute all tasks concurrently
    print(f"Starting async processing of {len(tasks)} images with max {max_concurrent} concurrent requests...")
    results = await asyncio.gather(*tasks)
    
    # Create and return DataFrame
    df = pd.DataFrame(results)
    return df

# Execute processing
if __name__ == "__main__":
    logs_file = "gemini/gemini_results3.txt"
    excel_file = "gemini/gemini_results3.xlsx"
    df = asyncio.run(process_images_from_file(logs_file=logs_file)) # logs file for results incase of error
    df.to_excel(excel_file, index=False) # results file for excel
    print(f"\nProcessed {len(df)} images, files saved in {predictions_dir} and logs saved in {logs_file} and excel saved in {excel_file}")
    print(df.head())