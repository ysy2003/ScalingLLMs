"""
Clean all HTML files in the merged results directory
Applies the correct cleaning logic to extract pure HTML
"""
import pandas as pd
import os
from clean_html_files import clean_html_content

print("="*60)
print("CLEANING MERGED HTML FILES")
print("="*60)

# Configuration
MERGED_DIR = "results_Qwen_merged"
RESULTS_FILE = f"{MERGED_DIR}/results/all_results.xlsx"
PREDICTIONS_DIR = f"{MERGED_DIR}/predictions"

# Load merged results
print(f"\nLoading merged results from {RESULTS_FILE}...")
df = pd.read_excel(RESULTS_FILE)
print(f"✓ Loaded {len(df)} samples")

# Clean each HTML file
print(f"\nCleaning HTML files...")
cleaned_count = 0
failed_count = 0

for _, row in df.iterrows():
    idx = row['number']
    original_text = row['response_text']
    
    # Apply cleaning
    cleaned_text = clean_html_content(original_text)
    
    # Save cleaned HTML
    html_file = f"{PREDICTIONS_DIR}/{idx}.html"
    try:
        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(cleaned_text)
        cleaned_count += 1
        
        if (idx + 1) % 50 == 0:
            print(f"  Cleaned {idx + 1} files...")
    except Exception as e:
        print(f"  ✗ Failed to clean {idx}.html: {e}")
        failed_count += 1

print("\n" + "="*60)
print("CLEANING COMPLETE!")
print("="*60)
print(f"\n✓ Cleaned {cleaned_count} HTML files")
if failed_count > 0:
    print(f"✗ Failed: {failed_count} files")
print(f"\nLocation: {PREDICTIONS_DIR}/")
print(f"All files now contain pure HTML (no prompts, no reasoning)")
