"""
CLIP Score calculation for Design2Code evaluation.

Compares rendered HTML predictions against reference images from the
HuggingFace dataset using CLIP embeddings cosine similarity.
"""

import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from torch.nn.functional import cosine_similarity
import asyncio
from playwright.async_api import async_playwright
from pathlib import Path
import pandas as pd
from datetime import datetime
from datasets import load_dataset
import io
import json

# --- Configuration ---
MODEL_NAME = "openai/clip-vit-base-patch32"
device = "cuda" if torch.cuda.is_available() else "cpu"

# Path to prediction folder (WebSight predictions)
PREDICTION_FOLDER = Path('/root/Design2code/results_WebSight/predictions')

# Path to save rendered images
RENDERED_IMGS_DIR = Path('/root/Design2code/results_WebSight/rendered_imgs')
RENDERED_IMGS_DIR.mkdir(parents=True, exist_ok=True)

# Path to save reference images from dataset
REF_IMGS_DIR = Path('/root/Design2code/results_WebSight/reference_imgs')
REF_IMGS_DIR.mkdir(parents=True, exist_ok=True)

# Output directory for results
OUTPUT_DIR = Path('/root/Design2code/results_WebSight')

# Sample range to evaluate (0-483 for full dataset)
SAMPLE_RANGE = range(0, 484)

# HuggingFace dataset
DATASET_NAME = "SALT-NLP/Design2Code-hf"

# --- CLIP Initialization ---
try:
    model = CLIPModel.from_pretrained(MODEL_NAME).to(device)
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    print(f"CLIP model loaded successfully ({MODEL_NAME}), device: {device}")
except Exception as e:
    print(f"Failed to load CLIP model: {e}")
    exit()


def calculate_clip_score(image_ref: Image.Image, image_rend: Image.Image) -> float:
    """
    Calculates the CLIP Score (Cosine similarity of image embeddings) between two images.

    Args:
        image_ref: Reference PIL Image
        image_rend: Rendered PIL Image

    Returns:
        CLIP score (cosine similarity) in range [-1, 1], typically [0, 1] for similar images
    """
    try:
        # Convert to RGB if needed
        img_ref = image_ref.convert("RGB")
        img_rend = image_rend.convert("RGB")
    except Exception as e:
        print(f"Error converting images: {e}")
        return 0.0

    # Preprocess images and convert them to tensors
    inputs = processor(images=[img_ref, img_rend], return_tensors="pt", padding=True)
    pixel_values = inputs.pixel_values.to(device)

    # Get image embeddings (Feature Vectors)
    with torch.no_grad():
        outputs = model.get_image_features(pixel_values=pixel_values)

    emb_ref = outputs[0].unsqueeze(0)
    emb_rend = outputs[1].unsqueeze(0)

    # Calculate Cosine Similarity
    similarity_tensor = cosine_similarity(emb_ref, emb_rend)

    # Return the result as a standard Python float
    clip_score = similarity_tensor.item()

    return clip_score


def calculate_clip_score_from_paths(image_path_ref: str, image_path_rend: str) -> float:
    """
    Calculates CLIP Score from file paths (for backward compatibility).
    """
    try:
        img_ref = Image.open(image_path_ref)
        img_rend = Image.open(image_path_rend)
        return calculate_clip_score(img_ref, img_rend)
    except FileNotFoundError as e:
        print(f"Error: File not found: {e}")
        return 0.0


async def render_html_to_image(page, html_content: str, output_path: str,
                                viewport_width: int = 1280, viewport_height: int = 720):
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

    # Set the HTML content
    await page.set_content(html_content)

    # Take a screenshot of the page and save it
    await page.screenshot(path=output_path)


def load_dataset_images(dataset, sample_indices):
    """
    Load reference images from the HuggingFace dataset.

    Args:
        dataset: HuggingFace dataset object
        sample_indices: List of sample indices to load

    Returns:
        Dict mapping index to PIL Image
    """
    images = {}
    for idx in sample_indices:
        if idx < len(dataset):
            sample = dataset[idx]
            # Dataset has 'image' field containing PIL Image
            if 'image' in sample:
                images[idx] = sample['image']
    return images


async def main():
    print('Loading HuggingFace dataset...')
    try:
        dataset = load_dataset(DATASET_NAME, split="train")
        dataset_size = len(dataset)
        print(f"Dataset loaded: {dataset_size} samples")
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return

    # Find available HTML predictions
    available_predictions = []
    for idx in SAMPLE_RANGE:
        html_file = PREDICTION_FOLDER / f"{idx}.html"
        if html_file.exists():
            available_predictions.append(idx)

    print(f"Found {len(available_predictions)} HTML predictions in range {SAMPLE_RANGE.start}-{SAMPLE_RANGE.stop-1}")

    if not available_predictions:
        print("No predictions found. Exiting.")
        return

    print(f"Starting CLIP score calculation...")

    # Initialize browser once for efficiency
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()

        all_scores = []
        processed_count = 0
        error_count = 0

        for idx in available_predictions:
            html_file = PREDICTION_FOLDER / f"{idx}.html"
            rendered_image_path = RENDERED_IMGS_DIR / f"{idx}.png"
            ref_image_path = REF_IMGS_DIR / f"{idx}.png"

            try:
                # Get reference image from dataset
                if idx >= len(dataset):
                    print(f"Warning: Index {idx} out of dataset range")
                    all_scores.append({
                        "sample_idx": idx,
                        "clip_score": None,
                        "error": "Index out of range"
                    })
                    error_count += 1
                    continue

                sample = dataset[idx]
                ref_image = sample['image']

                # Get reference image dimensions
                viewport_width, viewport_height = ref_image.size

                # Save reference image for inspection (optional)
                ref_image.save(ref_image_path)

                # Read HTML content
                with open(html_file, 'r', encoding='utf-8') as f:
                    html_content = f.read()

                # Render HTML to image with reference image dimensions
                await render_html_to_image(
                    page, html_content, str(rendered_image_path),
                    viewport_width, viewport_height
                )

                # Load rendered image
                rendered_image = Image.open(rendered_image_path)

                # Calculate CLIP score
                clip_score = calculate_clip_score(ref_image, rendered_image)

                all_scores.append({
                    "sample_idx": idx,
                    "clip_score": clip_score
                })

                processed_count += 1

                if processed_count % 10 == 0:
                    print(f"   Processed {processed_count}/{len(available_predictions)} files...")

            except Exception as e:
                print(f"Error processing sample {idx}: {e}")
                all_scores.append({
                    "sample_idx": idx,
                    "clip_score": None,
                    "error": str(e)
                })
                error_count += 1
                continue

        await browser.close()
        print('\nBrowser closed. Calculation complete.')

    # --- Save Results ---
    # Save to JSON (safer than Excel)
    json_filename = OUTPUT_DIR / "clip_scores.json"
    with open(json_filename, 'w') as f:
        json.dump(all_scores, f, indent=2)
    print(f"\nResults saved to: {json_filename}")

    # Also save to Excel
    df = pd.DataFrame(all_scores)
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    excel_filename = OUTPUT_DIR / f"clip_scores_{timestamp}.xlsx"
    df.to_excel(excel_filename, index=False)
    print(f"Results saved to: {excel_filename}")

    # --- Print Summary ---
    print('\n' + '='*60)
    print('CLIP Score Summary')
    print('='*60)

    # Filter out None scores for statistics
    valid_scores = [r['clip_score'] for r in all_scores if r['clip_score'] is not None]

    print(f"\n## Dataset Coverage")
    print(f"   Total Dataset Size:                  {dataset_size}")
    print(f"   Samples in Range ({SAMPLE_RANGE.start}-{SAMPLE_RANGE.stop-1}):          {len(SAMPLE_RANGE)}")
    print(f"   Predictions Found:                   {len(available_predictions)}")
    print(f"   Successfully Processed:              {processed_count}")
    print(f"   Errors:                              {error_count}")

    if valid_scores:
        avg_score = sum(valid_scores) / len(valid_scores)
        min_score = min(valid_scores)
        max_score = max(valid_scores)

        print(f"\n## CLIP Score Metrics")
        print(f"   Average CLIP Score:                  {avg_score:.4f} (range [0,1], higher is better)")
        print(f"   Minimum CLIP Score:                  {min_score:.4f}")
        print(f"   Maximum CLIP Score:                  {max_score:.4f}")

        # Calculate score distribution
        high_scores = sum(1 for s in valid_scores if s >= 0.8)
        medium_scores = sum(1 for s in valid_scores if 0.5 <= s < 0.8)
        low_scores = sum(1 for s in valid_scores if s < 0.5)

        print(f"\n## Score Distribution")
        print(f"   High (>=0.8):                        {high_scores} ({high_scores/len(valid_scores)*100:.1f}%)")
        print(f"   Medium (0.5-0.8):                    {medium_scores} ({medium_scores/len(valid_scores)*100:.1f}%)")
        print(f"   Low (<0.5):                          {low_scores} ({low_scores/len(valid_scores)*100:.1f}%)")
    else:
        print("\nNo valid scores to calculate statistics.")

    print('='*60)

    return all_scores


if __name__ == "__main__":
    asyncio.run(main())
