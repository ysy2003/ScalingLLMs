"""
Render Design2Code-18B HTML predictions to PNG images.

Usage:
    python render_design2code18b.py --predictions-dir results_Design2Code18B/predictions --output-dir results_Design2Code18B/rendered
"""

import argparse
import asyncio
from pathlib import Path
from tqdm import tqdm
from PIL import Image

try:
    from playwright.async_api import async_playwright
except ImportError:
    raise ImportError("Playwright not available. Install: pip install playwright && playwright install")

try:
    from datasets import load_dataset
except ImportError:
    raise ImportError("datasets not available. Install: pip install datasets")


async def render_html_to_png(predictions_dir: Path, output_dir: Path, samples: int):
    """Render HTML files to PNG images using Playwright."""

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset to get reference image sizes
    print("Loading dataset for reference image sizes...")
    dataset = load_dataset("SALT-NLP/Design2Code-hf", split="train")

    # Get HTML files to render
    html_files = sorted(predictions_dir.glob("*.html"))[:samples]

    if not html_files:
        print(f"No HTML files found in {predictions_dir}")
        return

    print(f"Found {len(html_files)} HTML files to render")

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()

        for html_path in tqdm(html_files, desc="Rendering"):
            idx = int(html_path.stem)
            output_path = output_dir / f"{idx}.png"

            # Skip if already exists
            if output_path.exists():
                continue

            # Get viewport size from dataset reference image
            try:
                ref_image = dataset[idx]['image']
                width, height = ref_image.size
            except Exception:
                width, height = 1280, 720

            # Read HTML content
            try:
                with open(html_path, 'r', encoding='utf-8') as f:
                    html_content = f.read()

                # Set viewport and render
                await page.set_viewport_size({"width": width, "height": height})
                await page.set_content(html_content, wait_until='networkidle')
                await page.screenshot(path=str(output_path))

            except Exception as e:
                print(f"\nError rendering {html_path.name}: {e}")
                continue

        await browser.close()

    # Count rendered files
    rendered_count = len(list(output_dir.glob("*.png")))
    print(f"\nRendered {rendered_count} PNG files to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Render Design2Code-18B predictions to PNG")
    parser.add_argument("--predictions-dir", type=str, required=True,
                        help="Directory containing HTML predictions")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Directory to save rendered PNG files")
    parser.add_argument("--samples", type=int, default=50,
                        help="Maximum number of samples to render")
    args = parser.parse_args()

    predictions_dir = Path(args.predictions_dir)
    output_dir = Path(args.output_dir)

    if not predictions_dir.exists():
        print(f"Error: Predictions directory not found: {predictions_dir}")
        return

    asyncio.run(render_html_to_png(predictions_dir, output_dir, args.samples))


if __name__ == "__main__":
    main()
