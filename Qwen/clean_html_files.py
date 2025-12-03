"""
Retroactively clean HTML files that were generated before the cleaning fix.
Removes chat markers and extracts HTML from markdown code blocks.
"""
import os
import sys

def clean_html_content(content):
    """Apply the same cleaning logic as in the test scripts"""
    cleaned_text = content

    # Remove chat markers (user/assistant)
    if "assistant" in cleaned_text:
        cleaned_text = cleaned_text.split("assistant", 1)[-1]

    # Remove <think>...</think> reasoning FIRST (before looking for code blocks)
    if "</think>" in cleaned_text:
        # Remove everything from start to </think>
        cleaned_text = cleaned_text.split("</think>", 1)[-1]

    # Extract from ```html ... ``` code block (now without <think> interference)
    if "```html" in cleaned_text:
        parts = cleaned_text.split("```html", 1)
        if len(parts) > 1:
            html_part = parts[1].split("```")[0]
            cleaned_text = html_part.strip()
    elif "```" in cleaned_text:
        # Fallback: extract from any code block
        parts = cleaned_text.split("```", 2)
        if len(parts) >= 3:
            cleaned_text = parts[1].strip()

    # Extract HTML from <!DOCTYPE onwards (in case there's no code block)
    if "<!DOCTYPE" in cleaned_text:
        cleaned_text = cleaned_text[cleaned_text.find("<!DOCTYPE"):]
    elif "<html" in cleaned_text.lower():
        # Fallback: extract from <html tag
        idx = cleaned_text.lower().find("<html")
        cleaned_text = cleaned_text[idx:]

    # Trim everything after </html>
    if "</html>" in cleaned_text:
        end_idx = cleaned_text.find("</html>") + len("</html>")
        cleaned_text = cleaned_text[:end_idx]

    return cleaned_text

def clean_html_files(start_idx, end_idx, predictions_dir="results_Qwen/predictions"):
    """Clean HTML files from start_idx to end_idx (inclusive)"""
    cleaned_count = 0
    failed_count = 0
    
    for idx in range(start_idx, end_idx + 1):
        html_file = os.path.join(predictions_dir, f"{idx}.html")
        
        if not os.path.exists(html_file):
            print(f"⚠ File not found: {html_file}")
            continue
        
        try:
            # Read original content
            with open(html_file, 'r', encoding='utf-8') as f:
                original_content = f.read()
            
            # Clean content
            cleaned_content = clean_html_content(original_content)
            
            # Only write if content changed
            if cleaned_content != original_content:
                with open(html_file, 'w', encoding='utf-8') as f:
                    f.write(cleaned_content)
                print(f"✓ Cleaned: {idx}.html")
                cleaned_count += 1
            else:
                print(f"- Skipped (already clean): {idx}.html")
        
        except Exception as e:
            print(f"✗ Failed to clean {idx}.html: {e}")
            failed_count += 1
    
    print(f"\n{'='*60}")
    print(f"SUMMARY:")
    print(f"  Cleaned: {cleaned_count}")
    print(f"  Failed: {failed_count}")
    print(f"  Total processed: {end_idx - start_idx + 1}")
    print(f"{'='*60}")

if __name__ == "__main__":
    # Test on files 0-10
    clean_html_files(0, 10)
