# remove ```html and ``` from predictions
import os
import re
from pathlib import Path
from typing import List

# --- Configuration (Ensure these paths match your evaluation script) ---
# Assuming your prediction results are located under 'gemini/results'
BASE_DIR = Path('gemini/results').resolve()

# List of prediction folders to clean
PREDICTION_FOLDERS: List[Path] = [
    BASE_DIR / 'gemini_predictions_perturbed',
    # BASE_DIR / 'gemini_predictions2', # Add more folders here if needed
]
# ---------------------------------------------------------------------

def clean_file(file_path: Path) -> bool:
    """
    Reads the HTML file, removes Markdown code block markers (e.g., ```html) 
    from the beginning and end, and overwrites the original file.
    
    Returns True if the file was modified, False otherwise.
    """
    try:
        # Read the entire file content
        content = file_path.read_text(encoding='utf-8')
        lines = content.strip().splitlines()
        
        if not lines:
            return False

        modified = False
        
        # 1. Check and clean the first line for the opening marker
        first_line = lines[0].strip()
        # Regex matches the start marker: ```html, ```js, ```, etc. (case-insensitive)
        # \s* matches optional whitespace
        if re.match(r'^\s*```(\w+)?\s*$', first_line, re.IGNORECASE):
            lines.pop(0)
            modified = True
        
        # 2. Check and clean the last line for the closing marker
        if lines:
            last_line = lines[-1].strip()
            # Match the closing ``` marker
            if last_line == '```':
                lines.pop()
                modified = True
                
        if modified:
            # Recombine the remaining lines and strip extra whitespace
            cleaned_content = '\n'.join(lines).strip()
            
            # Only write back if the content is not empty, preventing creation of empty files
            if cleaned_content:
                file_path.write_text(cleaned_content, encoding='utf-8')
                return True
            else:
                # If the content became empty after cleaning, handle it (e.g., delete or log)
                # For this script, we'll just skip writing and return False, but count as modified for logging purposes
                return True
                
        return False

    except Exception as e:
        # Print error if file processing fails
        print(f"❌ Error processing {file_path.name}: {e}")
        return False

def clean_all_predictions():
    """Main function: Iterates over all prediction folders and performs the cleaning."""
    print("🧹 Starting HTML Markdown code block cleaning...")
    
    total_files = 0
    cleaned_count = 0
    
    for folder in PREDICTION_FOLDERS:
        if not folder.is_dir():
            print(f"⚠️ Warning: Folder not found: {folder}")
            continue

        print(f"\nProcessing folder: {folder.name}")
        
        for file_path in folder.glob('*.html'):
            total_files += 1
            if clean_file(file_path):
                cleaned_count += 1

    print("\n--- Cleaning Summary ---")
    print(f"Total HTML files checked: {total_files}")
    print(f"Files where markers were successfully removed: {cleaned_count}")
    print("✅ Cleaning complete. You can now run the evaluation script.")

if __name__ == "__main__":
    clean_all_predictions()