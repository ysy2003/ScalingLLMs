import time
from typing import Optional
import pandas as pd
import sys
sys.path.insert(1, 'models/CogVLM')
from sat.model import AutoModel
import argparse
import torch
from sat.model.mixins import CachedAutoregressiveMixin
from sat.quantization.kernels import quantize
from sat.model import AutoModel
from utils.utils import chat, llama2_tokenizer, llama2_text_processor_inference, get_image_processor
from utils.models import CogAgentModel, CogVLMModel, FineTuneTestCogAgentModel
import os
import argparse

# Model name
model_name = "SALT-NLP/Design2Code-18B-v0"
model_path = "path/to/design2code-18b-v0"
predictions_dir = "models/design2code_18b_v0_predictions"

parser = argparse.ArgumentParser()
parser.add_argument('--temperature', type=float, default=0.5)
parser.add_argument('--repetition_penalty', type=float, default=1.1)
args = parser.parse_args()
args.bf16 = True
args.stream_chat = False
args.version = "chat"

if not os.path.exists(predictions_dir):
    try:
        os.makedirs(predictions_dir)
    except:
        pass

world_size = 1
model, model_args = FineTuneTestCogAgentModel.from_pretrained(
        model_path,
        args=argparse.Namespace(
        deepspeed=None,
        local_rank=0,
        rank=0,
        world_size=world_size,
        model_parallel_size=1,
        mode='inference',
        skip_init=True,
        use_gpu_initialization=True,
        device='cuda',
        bf16=True,
        fp16=None), overwrite_args={'model_parallel_size': world_size} if world_size != 1 else {})
model = model.eval()
model.add_mixin('auto-regressive', CachedAutoregressiveMixin())

language_processor_version = model_args.text_processor_version if 'text_processor_version' in model_args else args.version
print("[Language processor version]:", language_processor_version)
tokenizer = llama2_tokenizer("lmsys/vicuna-7b-v1.5", signal_type=language_processor_version)
image_processor = get_image_processor(model_args.eva_args["image_size"][0])
cross_image_processor = get_image_processor(model_args.cross_image_pix) if "cross_image_pix" in model_args else None
text_processor_infer = llama2_text_processor_inference(tokenizer, 2048, model.image_length)







def count_tokens(text: str) -> int:
    """
    Count tokens in text using the model's tokenizer
    """
    if not text:
        return 0
        
    token_ids = tokenizer.encode(text, add_special_tokens=False) 
    return len(token_ids)

def process_single_image(number: str, png_path: str, html_path: Optional[str]):
    """
    Process a single image and generate code
    
    Args:
        number: Image number identifier
        png_path: Local PNG image file path (e.g., Design2Code/{number}.png)
        html_path: Corresponding HTML file path (e.g., Design2Code/{number}.html)
    
    Returns:
        Dictionary with result data matching gemini.py format
    """
    print(f"Processing {png_path}...")
    
    

    # Check if image file exists
    if not os.path.exists(png_path):
        print(f"Image file not found: {png_path}")
        return {
            'png_uri': png_path,
            'html_uri': html_path,
            'response_text': None,
            'number': number,
            'error': f'Image file not found: {png_path}',
            'prompt_token_count': None,
            'candidates_token_count': None,
            'total_token_count': None,
            'latency': None
        }
    
    try:
        
        # Generate code
        start_time = time.time()
        with torch.no_grad():
            history = None
            cache_image = None
            # We use an empty string as the query
            query = ''
        
            response, history, cache_image = chat(
                png_path,
                model,
                text_processor_infer,
                image_processor,
                query,
                history=history,
                cross_img_processor=cross_image_processor,
                image=cache_image,
                max_length=4096,
                top_p=1.0,
                temperature=args.temperature,
                top_k=1,
                invalid_slices=text_processor_infer.invalid_slices,
                repetition_penalty=args.repetition_penalty,
                args=args
            )
    
        latency = time.time() - start_time

        
        generated_text = response
        with open(os.path.join(predictions_dir, png_path.split("/")[-1].replace(".png", ".html")), "w", encoding='utf-8') as f:
            f.write(generated_text)
        # count tokens
        processed_response = generated_text.replace(tokenizer.eos_token, '').strip()
        candidates_token_count = count_tokens(processed_response)
        bos_token_count=1
        prompt_token_count = bos_token_count+count_tokens(" [INST] " + query + " [/INST] ")+model.image_length+ 2 # 2 for the special tokens
        total_token_count = prompt_token_count + candidates_token_count


        return {
            'png_uri': png_path,
            'html_uri': html_path,
            'response_text': response,
            'number': number,
            'prompt_token_count': prompt_token_count,
            'candidates_token_count': candidates_token_count,
            'total_token_count': total_token_count,
            'latency': latency
        }
        
    except Exception as e:
        print(f"Error processing {png_path}: {str(e)}")
        
        return {
            'png_uri': png_path,
            'html_uri': html_path,
            'response_text': None,
            'number': number,
            'error': str(e),
            'prompt_token_count': None,
            'candidates_token_count': None,
            'total_token_count': None,
            'latency': None
        }

def process_images_from_file(n_images=None, data_dir="Design2Code"):
    """
    Scan Design2Code directory, process all PNG images, and return a DataFrame containing response.text and corresponding HTML URIs
    
    Args:
        n_images: Number of images to process (None for all)
        prompt: Prompt text file path
        data_dir: Local directory containing PNG and HTML files (default: "Design2Code")
    
    Returns:
        DataFrame with results matching gemini.py format
    """
    
    # Check if data directory exists
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    # Scan directory for PNG and HTML files
    png_paths = {}
    html_paths = {}
    
    # Get all files in the directory
    for filename in os.listdir(data_dir):
        file_path = os.path.join(data_dir, filename)
        
        # Skip if not a file
        if not os.path.isfile(file_path):
            continue
        
        # Extract number from filename
        if filename.endswith('.png'):
            number = filename.replace('.png', '')
            png_paths[number] = file_path
        elif filename.endswith('.html'):
            number = filename.replace('.html', '')
            html_paths[number] = file_path
    
    # Limit number of images if specified
    if n_images is not None:
        png_paths = {k: v for k, v in list(png_paths.items())[:n_images]}
    
    # Prepare list to store results
    results = []
    total_count = len(png_paths)
    
    # Process each image
    for i, (number, png_path) in enumerate(png_paths.items()):
        print(f"Processing image {i+1}/{total_count}: {number}")
        
        # Get corresponding HTML path
        html_path = html_paths.get(number, None)
        
        # Process image
        result = process_single_image(number, png_path, html_path)
        results.append(result)
        
        remaining = total_count - (i + 1)
        print(f"Completed {number}: {i+1}/{total_count} (remaining: {remaining})")
    
    # Create and return DataFrame
    df = pd.DataFrame(results)
    return df

# Execute processing
if __name__ == "__main__":
    df = process_images_from_file(n_images=2)
    df.to_excel("models/design2code_18b_v0_results.xlsx", index=False)
    print(f"\nProcessed {len(df)} images")
    print(df.head())
