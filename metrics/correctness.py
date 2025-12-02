# Required packages: 
# pip install playwright pandas openpyxl
#
# Also requires browser binaries:
# playwright install

import asyncio
from playwright.async_api import async_playwright
from pathlib import Path
import pandas as pd
import os

# --- Configuration ---

# Assumes folders are named 'gemini_predictions1', 'gemini_predictions2', etc.
BASE_DIR = Path('gemini/results').resolve()
PREDICTION_FOLDERS = [
    BASE_DIR / 'gemini_predictions1',
    # BASE_DIR / 'gemini_predictions2',
]

# Path to dataURI.txt file containing all test cases
DATAURI_FILE = Path('gemini/dataURI.txt')

# The name of the output Excel file.
OUTPUT_FILE = "metrics/evaluation/gemini_correctness_report.xlsx"

# -------------

async def check_file(browser, file_path: Path, sample_id: int):
    """
    Checks a single HTML file for rendering success and console errors.
    Now takes 'sample_id' to track which prediction batch it belongs to.
    """
    file_name = file_path.name
    context = None
    try:
        # Create a new, isolated browser context and page
        context = await browser.new_context()
        page = await context.new_page()

        errors = []
        render_success = False 

        # 1. Listen for 'console' events
        def handle_console(msg):
            if msg.type.lower() == 'error':
                errors.append(f"[Console Error]: {msg.text}")
        
        page.on('console', handle_console)

        # 2. Listen for 'pageerror' events
        def handle_page_error(err):
            errors.append(f"[Page Error]: {err.message}")

        page.on('pageerror', handle_page_error)

        # Load the HTML file
        file_url = f'file://{file_path.resolve()}'
        # Reduced timeout to 5s to speed up 3x processing if files are simple
        await page.goto(file_url, wait_until='load', timeout=10000) 

        # --- Metric: Render Success ---
        body_content = await page.evaluate("() => document.body.innerHTML")
        if body_content and len(body_content.strip()) > 0:
            render_success = True

        await context.close()

        return {
            "fileName": file_name,
            "sample_id": sample_id,  # Track which folder (1, 2, or 3)
            "renderSuccess": render_success,
            "errorCount": len(errors),
            "criticalErrorCount": 0,
            "errors": str(errors) # Convert list to string for Excel
        }

    except Exception as e:
        if context:
            await context.close()
        return {
            "fileName": file_name,
            "sample_id": sample_id,
            "renderSuccess": False,
            "errorCount": 0,
            "criticalErrorCount": 1,
            "errors": f"[Critical Load Error]: {e}"
        }

def load_test_cases_from_datauri(datauri_file: Path):
    """
    Load all HTML test case IDs from dataURI.txt file.
    Returns a set of file IDs (without .html extension).
    """
    test_cases = set()
    

    
    with open(datauri_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            # Extract HTML file names (e.g., "gs://dataset_design2code/10473.html" -> "10473")
            if line.endswith('.html'):
                file_name = os.path.basename(line)
                file_id = file_name.replace('.html', '')
                test_cases.add(file_id)
    
    return test_cases

async def main():
    print('📋 Loading test cases from dataURI.txt...')
    test_cases = load_test_cases_from_datauri(DATAURI_FILE)
    
    if not test_cases:
        print("❌ No test cases found in dataURI.txt. Exiting.")
        return
    
    print(f"✅ Found {len(test_cases)} test cases in dataURI.txt")
    
    print('🚀 Launching headless browser (Playwright)...')
    
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        
        all_results = []

        # --- Loop through each prediction folder ---
        for i, folder in enumerate(PREDICTION_FOLDERS):
            sample_id = i + 1 # 1, 2, 3
            print(f"\n📂 Processing Batch {sample_id}: {folder}")
            
            if not folder.exists():
                print(f"   ❌ Warning: Folder not found: {folder}")
                continue

            # Check each test case from dataURI.txt
            for file_id in test_cases:
                html_file = folder / f"{file_id}.html"
                
                if html_file.exists():
                    # File exists, check it with browser
                    result = await check_file(browser, html_file, sample_id)
                    all_results.append(result)
                else:
                    # File doesn't exist, mark as failed
                    all_results.append({
                        "fileName": f"{file_id}.html",
                        "sample_id": sample_id,
                        "renderSuccess": False,
                        "errorCount": 1,
                        "criticalErrorCount": 1,
                        "errors": f"[File Not Found]: {html_file.name} not found in {folder.name}"
                    })
            
            # Count how many files were found vs missing
            found_count = sum(1 for file_id in test_cases if (folder / f"{file_id}.html").exists())
            missing_count = len(test_cases) - found_count
            print(f"   📊 Found: {found_count}, Missing: {missing_count}")
            
        await browser.close()
        print('\n✅ Browser closed. Calculation complete.')


        df = pd.DataFrame(all_results)

        # Renaming for clarity
        df = df.rename(columns={
            'fileName': 'File Name',
            'sample_id': 'Sample ID',
            'renderSuccess': 'Passed',
            'errorCount': 'Error Count',
            'criticalErrorCount': 'Critical Error Count',
            'errors': 'Error Details'
        })

        # Ensure 'Passed' is boolean
        df['Passed'] = df['Passed'].astype(bool)

        # --- Calculate Pass@k Metrics ---
        print('\n--- 📊 Design2Code Pass@k Report ---')

        # 1. Pass@1: The global average accuracy across all predictions
        # Formula: Total Successes / Total Predictions
        pass_at_1 = df['Passed'].mean()

        # 2. Pass@3: For each unique file, did AT LEAST ONE sample pass?
        # Group by File Name, check if any sample in the group is True
        grouped = df.groupby('File Name')['Passed']
        
        # .max() on boolean acts as OR (True if any are True)
        file_level_pass = grouped.max() 
        
        total_unique_files = len(file_level_pass)

        
        # --- Display Metrics ---
        print(f"\n## Summary Statistics")
        print(f"   Total Unique Test Cases (from dataURI.txt): {total_unique_files}")
        print(f"   Total Predictions Evaluated:   {len(df)}")
        
        
        print(f"\n## 🏆 Metrics")
        print(f"   Pass@1 (Avg Accuracy):   {pass_at_1:.2%}")
        
        # --- Error Statistics ---
        print(f"\n## 📊 Error Statistics")
        print(f"   Error Count:")
        print(f"      Total:                 {df['Error Count'].sum()}")
        print(f"      Average:               {df['Error Count'].mean():.2f}")
        print(f"      Max:                    {df['Error Count'].max()}")
        print(f"      Files with errors:     {(df['Error Count'] > 0).sum()}")
        
        print(f"\n   Critical Error Count:")
        print(f"      Total:                 {df['Critical Error Count'].sum()}")
        print(f"      Average:               {df['Critical Error Count'].mean():.2f}")
        print(f"      Max:                    {df['Critical Error Count'].max()}")
        print(f"      Files with critical errors: {(df['Critical Error Count'] > 0).sum()}")

        # --- Save Detailed Report ---
        try:
            
            # We will save two sheets: Summary and Details
            with pd.ExcelWriter(OUTPUT_FILE) as writer:
                # Sheet 1: Raw Details (All rows)
                # Sort by Filename then Sample ID for readability
                df.sort_values(by=['File Name', 'Sample ID']).to_excel(writer, sheet_name='All Predictions', index=False)
                
                # Sheet 2: File Level Summary (Did file X pass at 3?)
                summary_df = df.groupby('File Name').agg({
                    'Passed': ['count', 'sum', 'max'], # Total runs, success count, did any pass
                    'Error Count': 'mean'
                }).reset_index()
                
                # Flatten MultiIndex columns
                summary_df.columns = ['File Name', 'Total Attempts', 'Successful Attempts', 'Pass@3 (Bool)', 'Avg Errors']
                summary_df.to_excel(writer, sheet_name='File Level Summary', index=False)

            print(f"\n✅ Detailed report saved to: {OUTPUT_FILE}")
            
        except Exception as e:
            print(f"\n❌ Error saving Excel: {e}")

if __name__ == "__main__":
    asyncio.run(main())