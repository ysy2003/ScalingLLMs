import sys
from bs4 import BeautifulSoup
from pathlib import Path
import os
import pandas as pd

# Add project root to sys.path before importing metrics
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import metrics.structural_alignment as structural_alignment

# --- Configuration ---
# All paths are relative to project_root to ensure they work regardless of where the script is run from
# Path to prediction folder
PREDICTION_FOLDER = project_root / 'design2code-18b-v0' / 'predictions'

# Path to reference HTML files (ground truth)
DESIGN2CODE_DIR = project_root / 'Design2Code'

# Path to save results
OUTPUT_DIR = project_root / 'design2code-18b-v0' / 'evaluation_results'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

class DOMAdapter:
    """
    Adapter class that converts BeautifulSoup nodes to a format
    that the structural_alignment module can understand
    (objects with tag, children, attrs attributes).
    """
    def __init__(self, bs_element):
        # BeautifulSoup uses .name, but the module needs .tag
        self.name = bs_element.name
        self.tag = bs_element.name
        
        # BeautifulSoup provides .attrs dictionary directly, reuse it
        self.attrs = bs_element.attrs
        
        # Recursively process child nodes
        # Note: We filter out NavigableString (pure text), only keep Tag nodes,
        # because structural_alignment mainly focuses on DOM structure tree comparison.
        self.children = []
        for child in bs_element.children:
            if hasattr(child, 'name') and child.name is not None:
                self.children.append(DOMAdapter(child))
    
    def find_all(self, recursive=True):
        """
        Mimics BeautifulSoup's find_all method.
        If recursive=False, returns only direct children.
        If recursive=True (default), returns all descendants including self.
        """
        if not recursive:
            # Return only direct children
            return self.children
        else:
            # Return all descendants including self
            result = [self]
            for child in self.children:
                result.extend(child.find_all(recursive=True))
            return result


def load_and_adapt(file_path):
    """Read HTML file and convert to adapted DOM tree"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            soup = BeautifulSoup(f, 'html.parser')
            
        # Start evaluation from body, if body doesn't exist then evaluate the entire soup
        root = soup.body if soup.body else soup
        return DOMAdapter(root)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None


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


def main():

    
    print('📋 Loading test cases from Design2Code folder...')
    test_cases = load_test_cases_from_design2code(DESIGN2CODE_DIR)
    
    if not test_cases:
        print("❌ No test cases found in Design2Code folder. Exiting.")
        return
    
    print(f"✅ Found {len(test_cases)} test cases in Design2Code folder")
    
    if not PREDICTION_FOLDER.exists():
        print(f"❌ Error: Prediction folder not found: {PREDICTION_FOLDER}")
        return
    
    print(f"🚀 Starting structural alignment evaluation for {PREDICTION_FOLDER}...")
    
    all_scores = []
    processed_count = 0
    skipped_count = 0
    error_count = 0
    
    # Process each test case
    for file_id in sorted(test_cases):
        ref_html = DESIGN2CODE_DIR / f"{file_id}.html"
        pred_html = PREDICTION_FOLDER / f"{file_id}.html"
        
        # Check if prediction HTML file exists
        if not pred_html.exists():
            all_scores.append({
                "file_id": file_id,
                "tree_edit_similarity": None,
                "semantic_html_ratio": None,
                "accessibility_score": None
            })
            skipped_count += 1
            continue
        
        # Check if reference HTML file exists
        if not ref_html.exists():
            print(f"⚠️  Warning: Reference HTML not found: {ref_html}")
            all_scores.append({
                "file_id": file_id,
                "tree_edit_similarity": None,
                "semantic_html_ratio": None,
                "accessibility_score": None
            })
            skipped_count += 1
            continue
        
        try:
            # Load and adapt both reference and prediction DOMs
            ref_dom = load_and_adapt(ref_html)
            pred_dom = load_and_adapt(pred_html)
            
            if not ref_dom or not pred_dom:
                print(f"⚠️  Warning: Failed to load DOM for {file_id}")
                all_scores.append({
                    "file_id": file_id,
                    "tree_edit_similarity": None,
                    "semantic_html_ratio": None,
                    "accessibility_score": None
                })
                error_count += 1
                continue
            
            # Calculate structural alignment scores
            scores = structural_alignment.compute_structural_alignment_scores(ref_dom, pred_dom)
            
            all_scores.append({
                "file_id": file_id,
                "tree_edit_similarity": scores.tree_edit_similarity,
                "semantic_html_ratio": scores.semantic_html_ratio,
                "accessibility_score": scores.accessibility_score
            })
            
            processed_count += 1
            
            if processed_count % 10 == 0:
                print(f"   Processed {processed_count} files...")
                
        except Exception as e:
            print(f"❌ Error processing {file_id}: {e}")
            all_scores.append({
                "file_id": file_id,
                "tree_edit_similarity": None,
                "semantic_html_ratio": None,
                "accessibility_score": None
            })
            error_count += 1
            continue
    
    # --- Create DataFrame and Save to Excel ---
    df = pd.DataFrame(all_scores)
    
    # Generate filename
    excel_filename = OUTPUT_DIR / "structural_alignment_scores.xlsx"
    
    # Save to Excel
    df.to_excel(excel_filename, index=False)
    print(f"\n✅ Results saved to: {excel_filename}")
    
    # --- Print Summary Statistics ---
    print('\n' + '='*60)
    print('📊 Structural Alignment Score Summary')
    print('='*60)
    
    if not all_scores:
        print("❌ No scores calculated. Check if files exist.")
        return
    
    # Filter out None scores for statistics
    valid_scores_tes = [result['tree_edit_similarity'] for result in all_scores if result['tree_edit_similarity'] is not None]
    valid_scores_sem = [result['semantic_html_ratio'] for result in all_scores if result['semantic_html_ratio'] is not None]
    valid_scores_acc = [result['accessibility_score'] for result in all_scores if result['accessibility_score'] is not None]
    
    print(f"\n## Summary Statistics")
    print(f"   Total Test Cases (from Design2Code): {len(test_cases)}")
    print(f"   Successfully Processed:              {processed_count}")
    print(f"   Skipped (file not found):            {skipped_count}")
    print(f"   Errors:                              {error_count}")
    
    if valid_scores_tes:
        avg_tes = sum(valid_scores_tes) / len(valid_scores_tes)
        min_tes = min(valid_scores_tes)
        max_tes = max(valid_scores_tes)
        
        print(f"\n## 🏆 Tree Edit Similarity Metrics (1.0 = identical)")
        print(f"   Average:                            {avg_tes:.4f}")
        print(f"   Minimum:                            {min_tes:.4f}")
        print(f"   Maximum:                            {max_tes:.4f}")
    
    if valid_scores_sem:
        avg_sem = sum(valid_scores_sem) / len(valid_scores_sem)
        min_sem = min(valid_scores_sem)
        max_sem = max(valid_scores_sem)
        
        print(f"\n## 🏆 Semantic HTML Ratio Metrics (Higher is better)")
        print(f"   Average:                            {avg_sem:.4f}")
        print(f"   Minimum:                            {min_sem:.4f}")
        print(f"   Maximum:                            {max_sem:.4f}")
    
    if valid_scores_acc:
        avg_acc = sum(valid_scores_acc) / len(valid_scores_acc)
        min_acc = min(valid_scores_acc)
        max_acc = max(valid_scores_acc)
        
        print(f"\n## 🏆 Accessibility Score Metrics (Higher is better)")
        print(f"   Average:                            {avg_acc:.4f}")
        print(f"   Minimum:                            {min_acc:.4f}")
        print(f"   Maximum:                            {max_acc:.4f}")
    
    if not valid_scores_tes and not valid_scores_sem and not valid_scores_acc:
        print("\n⚠️  No valid scores to calculate statistics.")
    
    print('='*60)


if __name__ == "__main__":
    main()
