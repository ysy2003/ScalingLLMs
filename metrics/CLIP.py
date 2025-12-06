import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from torch.nn.functional import cosine_similarity
import asyncio
from playwright.async_api import async_playwright
from pathlib import Path
import pandas as pd
from datetime import datetime

# --- Configuration ---
MODEL_NAME = "openai/clip-vit-base-patch32"
device = "cuda" if torch.cuda.is_available() else "cpu"

# Path to prediction folder
PREDICTION_FOLDER = Path('design2code-18b-v0/predictions')

# Path to reference images
DESIGN2CODE_DIR = Path('Design2Code')

# Path to save rendered images
RENDERED_IMGS_DIR = Path('design2code-18b-v0/rendered_imgs')
RENDERED_IMGS_DIR.mkdir(parents=True, exist_ok=True)

# --- CLIP Initialization ---
try:
    model = CLIPModel.from_pretrained(MODEL_NAME).to(device)
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    print(f"✅ CLIP model loaded successfully ({MODEL_NAME}), device used: {device}")
except Exception as e:
    print(f"❌ Failed to load CLIP model: {e}")
    exit()

def calculate_clip_score(image_path_ref: str, image_path_rend: str) -> float:
    """
    Calculates the CLIP Score (Cosine similarity of image embeddings) between two images.
    [Remaining function body is the same]
    """
    # [Function body remains the same as previous response, using image_path_ref and image_path_rend]
    try:
        # 1. Load images and convert to RGB
        img_ref = Image.open(image_path_ref).convert("RGB")
        img_rend = Image.open(image_path_rend).convert("RGB")
    except FileNotFoundError as e:
        print(f"❌ Error: File not found: {e}")
        return 0.0

    # 2. Preprocess images and convert them to tensors
    inputs = processor(images=[img_ref, img_rend], return_tensors="pt", padding=True)
    pixel_values = inputs.pixel_values.to(device)

    # 3. Get image embeddings (Feature Vectors)
    with torch.no_grad():
        outputs = model.get_image_features(pixel_values=pixel_values)
    
    emb_ref = outputs[0].unsqueeze(0)
    emb_rend = outputs[1].unsqueeze(0)

    # 4. Calculate Cosine Similarity
    similarity_tensor = cosine_similarity(emb_ref, emb_rend)
    
    # 5. Return the result as a standard Python float
    clip_score = similarity_tensor.item()
    
    return clip_score


async def render_html_to_image(page, html_content: str, output_path: str, viewport_width: int = 1280, viewport_height: int = 720):
    """
    Renders an HTML string into an image file using Playwright (Chromium headless browser).

    Args:
        page: Playwright page object (reused for efficiency)
        html_content: The HTML code string to be rendered.
        output_path: The file path to save the resulting image (e.g., 'rendered.png').
        viewport_width: The width of the virtual browser window.
        viewport_height: The height of the virtual browser window.
    """
    # Set the viewport size for consistent rendering
    await page.set_viewport_size({"width": viewport_width, "height": viewport_height})
    
    # Navigate to a blank page and set the HTML content
    # Using set_content ensures proper rendering of the HTML string
    await page.set_content(html_content)
    
    # Optionally wait for the network to be idle to ensure all resources (CSS/JS) are loaded
    # await page.wait_for_load_state("networkidle")
    
    # Take a screenshot of the page and save it
    await page.screenshot(path=output_path)


def load_test_cases_from_design2code(design2code_dir: Path):
    """
    Load all HTML test case IDs from Design2Code folder.
    Returns a set of file IDs (without .html extension).
    """
    test_cases = set()
    
    if not design2code_dir.exists():
        print(f"❌ Error: Design2Code folder not found at {design2code_dir}")
        return test_cases
    
    # Find all HTML files in the Design2Code directory
    for html_file in design2code_dir.glob("*.html"):
        # Extract file ID by removing .html extension
        file_id = html_file.stem  # stem gives filename without extension
        test_cases.add(file_id)
    
    return test_cases


async def main():
    print('📋 Loading test cases from Design2Code folder...')
    test_cases = load_test_cases_from_design2code(DESIGN2CODE_DIR)
    
    if not test_cases:
        print("❌ No test cases found in Design2Code folder. Exiting.")
        return
    
    print(f"✅ Found {len(test_cases)} test cases in Design2Code folder")
    
    if not PREDICTION_FOLDER.exists():
        print(f"❌ Error: Prediction folder not found: {PREDICTION_FOLDER}")
        return
    
    print(f"🚀 Starting CLIP score calculation for {PREDICTION_FOLDER}...")
    
    # Initialize browser once for efficiency
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        
        all_scores = []
        processed_count = 0
        skipped_count = 0
        error_count = 0
        
        # Process each test case - include all test cases, even if HTML doesn't exist
        for file_id in sorted(test_cases):
            html_file = PREDICTION_FOLDER / f"{file_id}.html"
            ref_image = DESIGN2CODE_DIR / f"{file_id}.png"
            rendered_image = RENDERED_IMGS_DIR / f"{file_id}.png"
            
            # Check if HTML file exists - if not, set score to None
            if not html_file.exists():
                all_scores.append({
                    "file_id": file_id,
                    "clip_score": None
                })
                skipped_count += 1
                continue
            
            # Check if reference image exists
            if not ref_image.exists():
                print(f"⚠️  Warning: Reference image not found: {ref_image}")
                raise Exception
 
            
            try:
                # Read reference image to get its dimensions
                with Image.open(ref_image) as img:
                    viewport_width, viewport_height = img.size
                
                # Read HTML content
                with open(html_file, 'r', encoding='utf-8') as f:
                    html_content = f.read()
                
                # Render HTML to image with reference image dimensions
                await render_html_to_image(page, html_content, str(rendered_image), viewport_width, viewport_height)
                
                # Calculate CLIP score
                clip_score = calculate_clip_score(str(ref_image), str(rendered_image))
                
                all_scores.append({
                    "file_id": file_id,
                    "clip_score": clip_score
                })
                
                processed_count += 1
                
                if processed_count % 10 == 0:
                    print(f"   Processed {processed_count} files...")
                    
            except Exception as e:
                print(f"❌ Error processing {file_id}: {e}")
                all_scores.append({
                    "file_id": file_id,
                    "clip_score": None
                })
                error_count += 1
                continue
        
        await browser.close()
        print('\n✅ Browser closed. Calculation complete.')
    
    # --- Create DataFrame and Save to Excel ---
    df = pd.DataFrame(all_scores)
    
    # Create output directory if it doesn't exist
    output_dir = Path('design2code-18b-v0/visual_fidelity')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate filename with timestamp
    excel_filename = output_dir / f"clip_scores.xlsx"
    
    # Save to Excel
    df.to_excel(excel_filename, index=False)
    print(f"\n✅ Results saved to: {excel_filename}")
    
    # --- Print Summary ---
    print('\n' + '='*60)
    print('📊 CLIP Score Summary')
    print('='*60)
    
    if not all_scores:
        print("❌ No scores calculated. Check if files exist.")
        return
    
    # Filter out None scores for statistics
    valid_scores = [result['clip_score'] for result in all_scores if result['clip_score'] is not None]
    
    print(f"\n## Summary Statistics")
    print(f"   Total Test Cases (from Design2Code): {len(test_cases)}")
    print(f"   Successfully Processed:              {processed_count}")
    print(f"   Skipped (file not found):            {skipped_count}")
    print(f"   Errors:                              {error_count}")
    print(f"   Missing HTML files:                  {skipped_count}")
    
    if valid_scores:
        avg_score = sum(valid_scores) / len(valid_scores)
        min_score = min(valid_scores)
        max_score = max(valid_scores)
        
        print(f"\n## 🏆 CLIP Score Metrics (for valid scores only)")
        print(f"   Average CLIP Score:                  {avg_score:.4f}")
        print(f"   Minimum CLIP Score:                  {min_score:.4f}")
        print(f"   Maximum CLIP Score:                  {max_score:.4f}")
        
        # Calculate score distribution
        high_scores = sum(1 for s in valid_scores if s >= 0.8)
        medium_scores = sum(1 for s in valid_scores if 0.5 <= s < 0.8)
        low_scores = sum(1 for s in valid_scores if s < 0.5)
        
        print(f"\n## Score Distribution")
        print(f"   High (≥0.8):                        {high_scores} ({high_scores/len(valid_scores)*100:.1f}%)")
        print(f"   Medium (0.5-0.8):                    {medium_scores} ({medium_scores/len(valid_scores)*100:.1f}%)")
        print(f"   Low (<0.5):                          {low_scores} ({low_scores/len(valid_scores)*100:.1f}%)")
    else:
        print("\n⚠️  No valid scores to calculate statistics.")
    
    print('='*60)
            
if __name__ == "__main__":
    # Python's asyncio is used to run the asynchronous Playwright function
    asyncio.run(main())