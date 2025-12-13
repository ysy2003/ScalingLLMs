import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
from typing import Optional
import importlib.metadata
import asyncio
import time
import json
from playwright.sync_api import sync_playwright
from metrics import correctness, efficiency, robustness, structural_alignment
from metrics.structural_alignment import compute_structural_alignment_scores
from google.cloud import storage

from agent.langchain_agent import AgentConfig, LangChainRepairAgent



# If 'packages_distributions' is unavailable, provide a fallback or handle the error gracefully
if not hasattr(importlib.metadata, 'packages_distributions'):
    print("⚠️ Warning: 'packages_distributions' is not available in 'importlib.metadata'.")
    # Add fallback logic here if necessary

def generate_html_css(agent, file_path, row_index: int = 0, html_column: Optional[str] = None):
    """
    Load initial HTML based on file type:
    - .html / .htm: Read as UTF-8 text
    - .xlsx / .xls: Read as spreadsheet, extract HTML string from specified row/column
    - Other extensions: Read as plain text file
    """
    print(f"Loading initial HTML from file: '{file_path}'...")

    ext = os.path.splitext(file_path)[1].lower()

    if ext in [".png", ".jpg", ".jpeg", ".webp"]:
        with open(file_path, "rb") as f:
            img_bytes = f.read()

        image_part = agent.build_multimodal_part(data=img_bytes, mime_type="image/png")
        prompt = "You are a design-to-code engine. Generate clean, responsive HTML+CSS for this landing page."

        response = agent.generate_content([image_part, prompt])
        return response if isinstance(response, str) else str(response)

    # 1) Pure HTML files, keep original logic
    if ext in [".html", ".htm"]:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                html = f.read()
            print(f"Loaded HTML file, length = {len(html)}")
            return html
        except FileNotFoundError:
            print(f"❌ Error: Test file not found at {file_path}")
            return "<html><body><h1>File not found</h1></body></html>"
        except UnicodeDecodeError as e:
            print(f"⚠️ UnicodeDecodeError while reading HTML file: {e}")
            # Fallback: Ignore illegal characters and read again
            with open(file_path, "r", errors="ignore") as f:
                html = f.read()
            print(f"Loaded HTML file with errors='ignore', length = {len(html)}")
            return html

    # 2) Excel files: Extract a segment of HTML / text from the spreadsheet
    if ext in [".xlsx", ".xls"]:
        try:
            df = pd.read_excel(file_path)
        except FileNotFoundError:
            print(f"❌ Error: Test file not found at {file_path}")
            return "<html><body><h1>File not found</h1></body></html>"

        if len(df) == 0:
            print(f"⚠️ Excel file is empty: {file_path}")
            return "<html><body><h1>Empty Excel file</h1></body></html>"

        # Fallback for row index
        if row_index >= len(df):
            print(f"⚠️ row_index={row_index} out of range, using last row")
            row_index = len(df) - 1

        # If html_column is not explicitly specified, guess a column by name
        if html_column is None:
            candidate_cols = ["html", "response", "output", "content", "answer"]
            for col in df.columns:
                if str(col).lower() in candidate_cols:
                    html_column = col
                    break
            if html_column is None:
                # If all else fails, use the first column
                html_column = df.columns[0]

        cell_value = df.iloc[row_index][html_column]
        html = cell_value if isinstance(cell_value, str) else str(cell_value)
        print(
            f"Loaded HTML from Excel: row={row_index}, column='{html_column}', length={len(html)}"
        )
        return html

    # 3) Other extensions: Read as plain text file
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            html = f.read()
        print(f"Loaded generic text file as HTML, length = {len(html)}")
        return html
    except FileNotFoundError:
        print(f"❌ Error: Test file not found at {file_path}")
        return "<html><body><h1>File not found</h1></body></html>"
    except UnicodeDecodeError:
        with open(file_path, "r", errors="ignore") as f:
            html = f.read()
        print(
            f"⚠️ UnicodeDecodeError on generic file, used errors='ignore'; length = {len(html)}"
        )
        return html

# Render HTML/CSS and capture screenshot
def render_and_capture(html, output_path):
    """
    Use Playwright to render HTML, capturing screenshot, console logs, and page errors.
    """
    errors = []
    render_success = False

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()

        # 1. Listen for 'console' events, capturing errors and warnings
        def handle_console(msg):
            if msg.type.lower() in ['error', 'warning']:
                errors.append(f"[Console {msg.type.upper()}]: {msg.text}")  # Fixed msg.text()

        page.on('console', handle_console)

        # 2. Listen for 'pageerror' events (uncaught JS exceptions)
        def handle_page_error(err):
            errors.append(f"[Page Error]: {err.message}")

        page.on('pageerror', handle_page_error)

        try:
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            # Load HTML using set_content
            page.set_content(html, wait_until='load')
            page.screenshot(path=output_path)

            # 3. Check if body is empty, indicating a critical rendering failure
            body_content = page.evaluate("() => document.body.innerHTML.trim()")
            if not body_content:
                errors.append("[Render Error]: Page body is empty, likely due to critical parsing error (e.g., unclosed tags).")
                render_success = False
            else:
                # Only consider rendering successful if the page has content
                render_success = True

        except Exception as e:
            render_success = False
            errors.append(f"[Critical Load Error]: {e}")
        finally:
            browser.close()

    return render_success, errors

def is_html_valid(html, render_success, captured_errors, metrics=None):
    """
    Basic HTML validity check:
    - Playwright rendering must succeed
    - No errors captured from console / pageerror
    - HTML itself has a certain length
    - Contains basic structure like <html> and <body>
    """
    if not render_success or captured_errors:
        return False

    lower = html.lower()
    if "<html" not in lower or "<body" not in lower:
        return False

    if metrics:
        if metrics.get("tree_edit_similarity", 0.0) < 0.9:
            return False
        if metrics.get("semantic_html_ratio", 0.0) < 0.05:
            return False

    return True

def compute_structural_alignment_scores(ref_dom, pred_dom):
    """
    Compute structural similarity between two DOM trees using tree edit distance.
    """
    from zss import simple_distance, Node

    def build_tree(dom):
        """Recursively build tree structure for edit distance calculation"""
        if not dom.name:
            return None
        root = Node(dom.name)
        for child in dom.find_all(recursive=False):
            child_node = build_tree(child)
            if child_node:
                root.addkid(child_node)
        return root

    ref_tree = build_tree(ref_dom)
    pred_tree = build_tree(pred_dom)

    if not ref_tree or not pred_tree:
        return 0.0  # Return 0 if either DOM tree is empty

    # Compute tree edit distance
    distance = simple_distance(ref_tree, pred_tree)
    max_size = max(len(ref_dom.find_all()), len(pred_dom.find_all()))
    similarity = 1 - (distance / max_size) if max_size > 0 else 0.0

    return similarity

def compute_semantic_ratio(dom):
    """
    Compute the ratio of semantic tags.
    """
    semantic_tags = {
        "header", "footer", "article", "section", "nav", "aside",
        "main", "h1", "h2", "h3"
    }
    body = dom.body or dom
    total_tags = len(body.find_all())
    semantic_tags_count = len([tag for tag in body.find_all() if tag.name in semantic_tags])
    return semantic_tags_count / total_tags if total_tags > 0 else 0.0

def compute_accessibility_score(dom):
    """
    Compute accessibility score based on multiple factors:
    - <img> tags with alt attributes
    - <form> tags with associated <label>
    - <a> tags with aria-label or title attributes
    """
    # Check <img> tags for alt attributes
    img_tags = dom.find_all("img")
    img_with_alt = len([img for img in img_tags if img.has_attr("alt")])
    img_score = img_with_alt / len(img_tags) if img_tags else 1.0

    # Check <form> tags for associated <label>
    form_tags = dom.find_all("form")
    labels = dom.find_all("label")
    form_score = len(labels) / len(form_tags) if form_tags else 1.0

    # Check <a> tags for aria-label or title attributes
    a_tags = dom.find_all("a")
    a_with_labels = len([a for a in a_tags if a.has_attr("aria-label") or a.has_attr("title")])
    a_score = a_with_labels / len(a_tags) if a_tags else 1.0

    # Combine scores (weighted average or simple average)
    total_score = (img_score + form_score + a_score) / 3
    return total_score

from bs4 import BeautifulSoup

def evaluate_metrics(screenshot_path, ground_truth_path):
    """
    Calculate quality metrics for HTML rendering.
    """
    try:
        # Load ground truth HTML
        with open(ground_truth_path, "r", encoding="utf-8", errors="replace") as f:
            ground_truth_html = f.read()
        ground_truth_dom = BeautifulSoup(ground_truth_html, "html.parser")

        # Load predicted HTML
        with open(screenshot_path, "r", encoding="utf-8", errors="replace") as f:
            predicted_html = f.read()
        predicted_dom = BeautifulSoup(predicted_html, "html.parser")

        # Debugging: Print DOM structure
        print("[DEBUG] Ground Truth DOM:", ground_truth_dom.prettify()[:500])
        print("[DEBUG] Predicted DOM:", predicted_dom.prettify()[:500])

        # Example metric calculations
        tree_edit_similarity = compute_structural_alignment_scores(ground_truth_dom, predicted_dom)
        semantic_html_ratio = compute_semantic_ratio(predicted_dom)
        accessibility_score = compute_accessibility_score(predicted_dom)

        # Debugging: Print calculated metrics
        print("[DEBUG] Tree Edit Similarity:", tree_edit_similarity)
        print("[DEBUG] Semantic HTML Ratio:", semantic_html_ratio)
        print("[DEBUG] Accessibility Score:", accessibility_score)

        return {
            "tree_edit_similarity": tree_edit_similarity,
            "semantic_html_ratio": semantic_html_ratio,
            "accessibility_score": accessibility_score,
        }
    except Exception as e:
        print(f"Error evaluating metrics: {e}")
        return {
            "tree_edit_similarity": 0.0,
            "semantic_html_ratio": 0.0,
            "accessibility_score": 0.0,
        }


def extract_html_from_response(cell_value: str) -> str:
    if not isinstance(cell_value, str):
        cell_value = str(cell_value or "")

    text = cell_value.strip()

    if len(text) >= 2 and text[0] == text[-1] == '"':
        text = text[1:-1]

    lower = text.lower()
    if "```html" in lower:
        start = lower.index("```html") + len("```html")
        text = text[start:]
        end_idx = text.find("```")
        if end_idx != -1:
            text = text[:end_idx]

    return text.strip()

def download_html_from_gcs(gcs_uri: str, local_path: str):
    """
    Download an HTML file from Google Cloud Storage to a local path.
    """
    try:
        client = storage.Client()
        bucket_name, blob_name = gcs_uri.replace("gs://", "").split("/", 1)
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        blob.download_to_filename(local_path)
        print(f"Downloaded {gcs_uri} to {local_path}")
    except Exception as e:
        print(f"❌ Error downloading {gcs_uri}: {e}")

def process_excel_file(agent, excel_path, max_files, max_attempts=3):
    """
    Process rows in an Excel file for repair.

    Args:
        model: The model used for repairing HTML.
        excel_path: Path to the Excel file.
        max_files: Maximum number of rows to process.
        max_attempts: Maximum number of repair attempts per row.
    """
    try:
        df = pd.read_excel(excel_path)
        if df.empty:
            print(f"❌ Error: Excel file is empty: {excel_path}")
            return

        # Limit the number of rows to process based on max_files
        rows_to_process = min(len(df), max_files)
        print(f"Processing {rows_to_process} rows from Excel file: {excel_path}")

        # Create an output directory for this Excel file
        base_output_dir = "agent/output"
        excel_file_name = os.path.splitext(os.path.basename(excel_path))[0]
        output_dir = os.path.join(base_output_dir, excel_file_name)
        os.makedirs(output_dir, exist_ok=True)

        PNG_COL = "png_uri"
        HTML_URI_COL = "html_uri"
        RESP_COL = "response_text"

        for i in range(rows_to_process):
            row = df.iloc[i]
            print(f"\n=== Processing Row {i + 1}/{rows_to_process} ===")

            # Extract columns
            png_uri = str(row[PNG_COL]) if PNG_COL in df.columns else None
            html_uri = str(row[HTML_URI_COL]) if HTML_URI_COL in df.columns else None
            raw_response = row[RESP_COL] if RESP_COL in df.columns else ""

            # Extract initial HTML from response_text
            html_content = extract_html_from_response(raw_response)

            # Write the initial HTML to a temporary file
            temp_html_path = os.path.join(output_dir, f"temp_row_{i + 1}.html")
            with open(temp_html_path, "w", encoding="utf-8") as temp_file:
                temp_file.write(html_content)

            # Download html_uri as ground truth
            ground_truth_path = None
            if html_uri:
                ground_truth_path = os.path.join(output_dir, f"gt_row_{i + 1}.html")
                download_html_from_gcs(html_uri, ground_truth_path)

            # Use temp_html as the starting point and ground_truth_path as the reference
            metrics, logs = repair_loop(
                agent,
                temp_html_path,
                ground_truth_html_path=ground_truth_path,
                max_attempts=max_attempts,
            )

            # Save logs
            log_path = os.path.join(output_dir, f"repair_log_row_{i + 1}.json")
            with open(log_path, "w", encoding="utf-8") as log_file:
                json.dump(logs, log_file, indent=4)

            print(f"Metrics for Row {i + 1}: {metrics}")

    except Exception as e:
        print(f"❌ Error processing Excel file: {e}")

def repair_loop(agent, html_file_path, ground_truth_html_path=None, max_attempts: int = 3):
    """
    Perform the repair loop for a given HTML file.

    Args:
        model: The model used for repairing HTML.
        html_file_path: Path to the initial HTML file.
        ground_truth_html_path: Path to the ground truth HTML file.
        max_attempts: Maximum number of repair attempts.
    """
    # Create a unique folder for this file's repair attempts
    base_output_dir = "agent/output"
    file_name = os.path.splitext(os.path.basename(html_file_path))[0]
    output_dir = os.path.join(base_output_dir, file_name)
    os.makedirs(output_dir, exist_ok=True)

    # Load the initial HTML (can be .html or .xlsx)
    html = generate_html_css(agent, html_file_path)

    # Print a snippet of the HTML for debugging purposes
    print("Initial HTML snippet (first 300 chars):")
    print(html[:300].replace("\n", "\\n"))

    logs = []
    metrics = {}

    for attempt in range(max_attempts + 1):  # attempt = 0 is the initial version
        print(f"\n--- Attempt {attempt}/{max_attempts} ---")

        # Write the current HTML to a .html file for metrics evaluation
        html_attempt_path = os.path.join(output_dir, f"output_attempt_{attempt}.html")
        with open(html_attempt_path, "w", encoding="utf-8") as f:
            f.write(html)

        # Render the HTML to a PNG (for visual debugging and JS error capture)
        screenshot_path = os.path.join(output_dir, f"output_attempt_{attempt}.png")
        render_success, captured_errors = render_and_capture(html, screenshot_path)
        print(f"Render Success: {render_success}, Errors Found: {len(captured_errors)}")

        # Evaluate metrics using the ground truth HTML
        metrics = evaluate_metrics(
            screenshot_path=html_attempt_path,
            ground_truth_path=ground_truth_html_path or html_file_path,
        )

        logs.append(
            {
                "attempt": attempt,
                "render_success": render_success,
                "errors": captured_errors,
                "metrics": metrics,
                "html_content": html,
            }
        )

        if is_html_valid(html, render_success, captured_errors, metrics):
            print("\n✅ HTML passes validity checks. Exiting repair loop.")
            break

        if attempt >= max_attempts:
            print("\n⚠️ Reached maximum attempts without obtaining valid HTML.")
            break

        html = agent.repair_html(html, captured_errors, metrics)
        print(f"Length of repaired HTML: {len(html)}")

    # Save logs to the output directory
    log_path = os.path.join(output_dir, "repair_log.json")
    with open(log_path, "w", encoding="utf-8") as log_file:
        json.dump(logs, log_file, indent=4)

    return metrics, logs

if __name__ == "__main__":
    # --- Load configuration from YAML file ---
    import yaml
    with open("agent/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    GCP_PROJECT_ID = config["gcp"]["project_id"]
    GCP_REGION = config["gcp"]["region"]
    MODEL_ID = config["model"]["id"]
    TEST_FILE_PATH = config["test_file_path"]
    MAX_FILES = config.get("max_files", 10)  # Default to 10 if not specified

    # Ensure TEST_FILE_PATH is a valid file
    if not os.path.isfile(TEST_FILE_PATH):
        print(f"❌ Error: TEST_FILE_PATH is not a valid file: {TEST_FILE_PATH}")
        sys.exit(1)

    # --- Validate configuration ---
    if not all([GCP_PROJECT_ID, GCP_REGION, MODEL_ID, TEST_FILE_PATH]):
        print("❌ Error: Missing one or more required configuration values.")
        print("Please ensure project_id, region, model_id, and test_file_path are set in config.yaml.")
        sys.exit(1)

    print(f"--- Configuration Loaded ---")
    print(f"Project: {GCP_PROJECT_ID}, Region: {GCP_REGION}")
    print(f"Model: {MODEL_ID}")
    print(f"Test File Path: {TEST_FILE_PATH}")
    print(f"Max Files: {MAX_FILES}")
    print(f"----------------------------")

    agent_config = AgentConfig.from_yaml(config)
    agent = LangChainRepairAgent(agent_config)

    # Process the Excel file
    process_excel_file(agent, TEST_FILE_PATH, MAX_FILES)