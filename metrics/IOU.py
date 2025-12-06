import asyncio
from playwright.async_api import async_playwright
from pathlib import Path
import pandas as pd

# --- Configuration ---
# Path to prediction folder
# PREDICTION_FOLDER = Path('design2code-18b-v0/predictions')
PREDICTION_FOLDER = Path('gemini/results/gemini_predictions1')

# Path to reference HTML files (ground truth)
DESIGN2CODE_DIR = Path('Design2Code')

# Path to save results
OUTPUT_DIR = Path('gemini/evaluation_results')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DEBUG_DIR = OUTPUT_DIR / "debug_screenshots"
DEBUG_DIR.mkdir(parents=True, exist_ok=True)

class LayoutBenchmark:
    def __init__(self):
        self.browser = None

    async def start(self):
        """Initializes the headless browser."""
        p = await async_playwright().start()
        # Launch Chromium headless browser
        self.browser = await p.chromium.launch(headless=True)
        self.playwright = p

    async def stop(self):
        """Closes the headless browser."""
        await self.browser.close()
        if self.playwright:
            await self.playwright.stop()


    async def get_element_bboxes(self, html_content, file_id, type_prefix):
        """
        Extracts bounding boxes. 
        1. Returns a LIST instead of Dict to prevent overwriting duplicate text.
        2. Takes screenshots for debugging.
        """
        page = await self.browser.new_page()
        await page.set_viewport_size({"width": 1280, "height": 800})
        

        try:
            await page.set_content(html_content, wait_until='networkidle', timeout=5000)
        except Exception:
            pass

        await page.add_style_tag(content="body { margin: 0; padding: 0; }")

        # await page.screenshot(path=DEBUG_DIR / f"{file_id}_{type_prefix}.png", full_page=True)

        elements_data = await page.evaluate('''() => {
            const results = [];
            
            const allElements = document.querySelectorAll('*');
            allElements.forEach((el) => {
                const isVisible = el.offsetWidth > 0 && el.offsetHeight > 0;
                const hasText = el.innerText && el.innerText.trim().length > 0;
                
                const validTag = !['SCRIPT', 'STYLE', 'NOSCRIPT'].includes(el.tagName);
                

                const isLeaf = el.children.length === 0;

                if (isVisible && hasText && isLeaf && validTag) {
                    const rect = el.getBoundingClientRect();
                    results.push({
                        text: el.innerText.trim(),
                        x: rect.x + window.scrollX,
                        y: rect.y + window.scrollY,
                        width: rect.width,
                        height: rect.height
                    });
                }
            });
            return results;
        }''')
        
        await page.close()
        return elements_data

    def compute_iou(self, box1, box2):
        """Calculates the Intersection over Union (IoU) of two bounding boxes."""
        # box: {x, y, width, height}
        
        # Convert to (x1, y1, x2, y2)
        b1_x1, b1_y1 = box1['x'], box1['y']
        b1_x2, b1_y2 = b1_x1 + box1['width'], b1_y1 + box1['height']
        
        b2_x1, b2_y1 = box2['x'], box2['y']
        b2_x2, b2_y2 = b2_x1 + box2['width'], b2_y1 + box2['height']

        # Calculate Intersection coordinates
        inter_x1 = max(b1_x1, b2_x1)
        inter_y1 = max(b1_y1, b2_y1)
        inter_x2 = min(b1_x2, b2_x2)
        inter_y2 = min(b1_y2, b2_y2)

        # Calculate Intersection Area
        inter_width = max(0, inter_x2 - inter_x1)
        inter_height = max(0, inter_y2 - inter_y1)
        inter_area = inter_width * inter_height

        # Calculate Union Area
        b1_area = box1['width'] * box1['height']
        b2_area = box2['width'] * box2['height']
        union_area = b1_area + b2_area - inter_area

        if union_area == 0: return 0.0
        return inter_area / union_area

    def compare_layouts(self, gt_list, pred_list):
        """
        Robust Layout Similarity: Spatial Greedy Matching.
        For each GT box, find the Pred box with the highest IoU.
        """
        if not gt_list: return 0.0
        if not pred_list: return 0.0

        total_max_iou = 0.0
        

        
        for gt_box in gt_list:
            max_iou_for_this_element = 0.0
            
            for pred_box in pred_list:

                
                iou = self.compute_iou(gt_box, pred_box)
                if iou > max_iou_for_this_element:
                    max_iou_for_this_element = iou
            
            total_max_iou += max_iou_for_this_element

        return total_max_iou / len(gt_list)


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
    
    print(f"🚀 Starting IOU score calculation for {PREDICTION_FOLDER}...")
    
    # Initialize browser and benchmark
    benchmark = LayoutBenchmark()
    await benchmark.start()
    
    all_scores = []
    processed_count = 0
    skipped_count = 0
    error_count = 0
    
    # Process each test case
    for file_id in sorted(test_cases):
        pred_html_file = PREDICTION_FOLDER / f"{file_id}.html"
        gt_html_file = DESIGN2CODE_DIR / f"{file_id}.html"
        
        # Check if prediction HTML file exists
        if not pred_html_file.exists():
            all_scores.append({
                "file_id": file_id,
                "iou_score": None
            })
            skipped_count += 1
            continue
        
        # Check if ground truth HTML file exists
        if not gt_html_file.exists():
            print(f"⚠️  Warning: Ground truth HTML not found: {gt_html_file}")
            all_scores.append({
                "file_id": file_id,
                "iou_score": None
            })
            skipped_count += 1
            continue
        
        try:
            # Read HTML content
            with open(pred_html_file, 'r', encoding='utf-8') as f:
                pred_html_content = f.read()
            
            with open(gt_html_file, 'r', encoding='utf-8') as f:
                gt_html_content = f.read()
            
            # Extract bounding boxes from both HTML files
            gt_boxes = await benchmark.get_element_bboxes(gt_html_content, file_id, "GT")
            pred_boxes = await benchmark.get_element_bboxes(pred_html_content, file_id, "PRED")
            
            # Calculate IOU score
            iou_score = benchmark.compare_layouts(gt_boxes, pred_boxes)
            
            all_scores.append({
                "file_id": file_id,
                "iou_score": iou_score
            })
            
            processed_count += 1
            
            if processed_count % 10 == 0:
                print(f"   Processed {processed_count} files...")
                
        except Exception as e:
            print(f"❌ Error processing {file_id}: {e}")
            all_scores.append({
                "file_id": file_id,
                "iou_score": None
            })
            error_count += 1
            continue
    
    await benchmark.stop()
    print('\n✅ Browser closed. Calculation complete.')
    
    # --- Create DataFrame and Save to Excel ---
    df = pd.DataFrame(all_scores)
    
    # Generate filename
    excel_filename = OUTPUT_DIR / "iou_scores.xlsx"
    
    # Save to Excel
    df.to_excel(excel_filename, index=False)
    print(f"\n✅ Results saved to: {excel_filename}")
    
    # --- Print Summary Statistics ---
    print('\n' + '='*60)
    print('📊 IOU Score Summary')
    print('='*60)
    
    if not all_scores:
        print("❌ No scores calculated. Check if files exist.")
        return
    
    # Filter out None scores for statistics
    valid_scores = [result['iou_score'] for result in all_scores if result['iou_score'] is not None]
    
    print(f"\n## Summary Statistics")
    print(f"   Total Test Cases (from Design2Code): {len(test_cases)}")
    print(f"   Successfully Processed:              {processed_count}")
    print(f"   Skipped (file not found):            {skipped_count}")
    print(f"   Errors:                              {error_count}")
    
    if valid_scores:
        avg_score = sum(valid_scores) / len(valid_scores)
        min_score = min(valid_scores)
        max_score = max(valid_scores)
        
        print(f"\n## 🏆 IOU Score Metrics (for valid scores only)")
        print(f"   Average IOU Score:                  {avg_score:.4f}")
        print(f"   Minimum IOU Score:                  {min_score:.4f}")
        print(f"   Maximum IOU Score:                  {max_score:.4f}")
        
        # Calculate score distribution
        high_scores = sum(1 for s in valid_scores if s >= 0.8)
        medium_scores = sum(1 for s in valid_scores if 0.5 <= s < 0.8)
        low_scores = sum(1 for s in valid_scores if s < 0.5)
        
        print(f"\n## Score Distribution")
        print(f"   High (≥0.8):                        {high_scores} ({high_scores/len(valid_scores)*100:.1f}%)")
        print(f"   Medium (0.5-0.8):                   {medium_scores} ({medium_scores/len(valid_scores)*100:.1f}%)")
        print(f"   Low (<0.5):                         {low_scores} ({low_scores/len(valid_scores)*100:.1f}%)")
    else:
        print("\n⚠️  No valid scores to calculate statistics.")
    
    print('='*60)


if __name__ == "__main__":
    # Python's asyncio is used to run the asynchronous Playwright function
    asyncio.run(main())