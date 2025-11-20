# Required packages: 
# pip install playwright
# pip install pandas
#
# Also requires browser binaries to be installed:
# playwright install

import asyncio
from playwright.async_api import async_playwright
from pathlib import Path
import os
import sys
import pandas as pd

# --- Configuration ---

# The path to your predictions folder.

PREDICTIONS_FOLDER = Path('gemini/results/gemini_predictions').resolve()

# The name of the output CSV file.
OUTPUT_FILE = "metrics/correctness_metrics_report.xlsx"

# -------------

async def check_file(browser, file_path: Path):
    """
    Checks a single HTML file for rendering success and console errors.
    
    Args:
        browser: The Playwright browser instance.
        file_path: The Path object pointing to the HTML file.

    Returns:
        A dictionary containing the analysis results for the file.
    """
    file_name = file_path.name
    context = None
    try:
        # Create a new, isolated browser context and page for each file.
        context = await browser.new_context()
        page = await context.new_page()

        errors = []
        render_success = False # Default to False

        # --- CRITICAL: Set up listeners *before* loading the page ---
        
        # 1. Listen for 'console' events (e.g., console.error)
        def handle_console(msg):
            if msg.type.lower() == 'error':
                errors.append(f"[Console Error]: {msg.text}")
        
        page.on('console', handle_console)

        # 2. Listen for 'pageerror' events (e.g., unhandled JavaScript exceptions)
        def handle_page_error(err):
            errors.append(f"[Page Error]: {err.message}")

        page.on('pageerror', handle_page_error)


        # Load the HTML file using the file:// protocol
        file_url = f'file://{file_path.resolve()}'
        await page.goto(file_url, wait_until='load') # Wait for the 'load' event

        # --- Metric 1: Render Success ---
        # Our definition: The page loaded and the <body> tag is not empty.
        body_content = await page.evaluate("() => document.body.innerHTML")
        if body_content and len(body_content.strip()) > 0:
            render_success = True

        # Clean up the context and page
        await context.close()

        return {
            "fileName": file_name,
            "renderSuccess": render_success,
            "errorCount": len(errors),
            "criticalErrorCount": 0,
            "errors": errors
        }

    except Exception as e:
        # This catches critical load errors (e.g., completely broken HTML)
        if context:
            await context.close() # Ensure cleanup
        return {
            "fileName": file_name,
            "renderSuccess": False, # Explicit render failure
            "errorCount": 1,        # Count this as one critical error
            "criticalErrorCount": 1,
            "errors": [f"[Critical Load Error]: {e}"]
        }

async def main():
    """
    Main function to run the analysis.
    """
    print('🚀 Launching headless browser (Playwright)...')
    
    async with async_playwright() as p:
        # Launch the Chromium browser
        browser = await p.chromium.launch(headless=True)



        # 2. Find all .html files in the folder
        print(f"📂 Reading files from: {PREDICTIONS_FOLDER}")
        files = list(PREDICTIONS_FOLDER.glob('*.html'))

        if not files:
            print(f"❌ Error: No .html files found in the specified folder.")
            await browser.close()
            return

        print(f"🔍 Found {len(files)} HTML files. Starting analysis...")

        # 3. Run the analysis for all files concurrently
        tasks = [check_file(browser, file) for file in files]
        results = await asyncio.gather(*tasks)
        
        print("...analysis complete.")

        await browser.close()
        print('✅ Browser closed.')

        # --- 4. Convert results to DataFrame ---
        if not results:
            print("No results to process.")
            return

        df = pd.DataFrame(results)

        # Rename columns to be more descriptive (as requested)
        df = df.rename(columns={
            'fileName': 'fileName',
            'renderSuccess': 'Render Success',
            'errorCount': 'DOM/Console Error Count',
            'criticalErrorCount': 'Critical Error Count',
            'errors': 'Errors Report'
        })



        # Re-order columns
        df = df[['fileName', 'Render Success', 'DOM/Console Error Count', 'Critical Error Count', 'Errors Report']]

        # --- 5. Print Summary and DataFrame ---
        print('\n--- 📊 Aggregate Metrics Report ---')

        total_files = len(df)
        
        # Metric 1: Render Success Rate (Aggregate)
        successful_renders = df['Render Success'].sum()
        render_success_rate = (successful_renders / total_files) * 100
        print(f"\n## 1. Render Success Rate")
        print(f"   {successful_renders} / {total_files} files rendered successfully (<body> not empty)")
        print(f"   Rate: {render_success_rate:.2f}%")

        # Metric 2: DOM/Console Error Count (Aggregate)
        total_errors = df['DOM/Console Error Count'].sum()
        avg_errors = df['DOM/Console Error Count'].mean()
        print(f"\n## 2. DOM/Console Error Count")
        print(f"   Total Errors Found: {total_errors}")
        print(f"   Average Errors per File: {avg_errors:.2f}")

        # Metric 3: Critical Error Count (Aggregate)
        total_critical_errors = df['Critical Error Count'].sum()
        avg_critical_errors = df['Critical Error Count'].mean()
        print(f"\n## 3. Critical Error Count")
        print(f"   Total Critical Errors Found: {total_critical_errors}")
        print(f"   Average Critical Errors per File: {avg_critical_errors:.2f}")

        # --- 6. Save to excel ---
        try:
            df.to_excel(OUTPUT_FILE, index=False)
            print(f"\n✅ Report successfully saved to:")
            print(f"   {OUTPUT_FILE}")
        except Exception as e:
            print(f"\n❌ Error saving report to CSV: {e}")


# --- Main execution block ---
if __name__ == "__main__":
        
    asyncio.run(main())