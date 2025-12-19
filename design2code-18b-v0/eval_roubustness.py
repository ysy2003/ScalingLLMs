from PIL import Image, ImageEnhance, ImageFilter
from pathlib import Path
import os
import shutil
import random

def perturb_image(img: Image.Image, strength: float = 0.1) -> Image.Image:
    """
    Apply slight perturbations to UI screenshots.
    Intensity ~ [0, 0.3]; higher values indicate stronger perturbations.
    """
    w, h = img.size

    # 1) Slight random crop + resize back to original size (simulate screenshot edge errors)
    max_crop = int(min(w, h) * strength)
    left   = random.randint(0, max_crop)
    top    = random.randint(0, max_crop)
    right  = w - random.randint(0, max_crop)
    bottom = h - random.randint(0, max_crop)
    img = img.crop((left, top, right, bottom)).resize((w, h), Image.BICUBIC)

    # 2) Brightness and contrast jitter
    b_factor = 1.0 + random.uniform(-strength, strength)
    c_factor = 1.0 + random.uniform(-strength, strength)
    img = ImageEnhance.Brightness(img).enhance(b_factor)
    img = ImageEnhance.Contrast(img).enhance(c_factor)

    # 3) Slight Gaussian blur 
    if strength > 0:
        radius = strength * 2.0
        img = img.filter(ImageFilter.GaussianBlur(radius=radius))

    return img


def load_test_cases_from_datauri(datauri_file: Path, n_cases: int = 50):
    """
    Load the first n_cases testcase IDs from dataURI.txt file.
    Reference the reading method in metrics/correctness.py.
    Returns a list containing testcase IDs.
    """
    test_cases = []
    
    with open(datauri_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            # Only extract IDs corresponding to .png files (avoid duplicates)
            if line.endswith('.png'):
                file_name = os.path.basename(line)
                file_id = file_name.replace('.png', '')
                if file_id not in test_cases:
                    test_cases.append(file_id)
                    if len(test_cases) >= n_cases:
                        break
    
    return test_cases


def main():
    """
    Main function:
    1. Read the first 50 testcases from dataURI.txt
    2. Copy corresponding png files from Design2Code/ directory to clean_imgs/
    3. Run perturb_image on each image
    4. Save results to perturbed_imgs/
    """
    # Configure paths
    BASE_DIR = Path(__file__).resolve().parent
    PROJECT_ROOT = BASE_DIR.parent
    
    DATAURI_FILE = PROJECT_ROOT / 'gemini' / 'dataURI.txt'
    DESIGN2CODE_DIR = PROJECT_ROOT / 'Design2Code'
    CLEAN_IMGS_DIR = BASE_DIR / 'clean_imgs'
    PERTURBED_IMGS_DIR = BASE_DIR / 'perturbed_imgs'
    
    # Create output directories
    CLEAN_IMGS_DIR.mkdir(parents=True, exist_ok=True)
    PERTURBED_IMGS_DIR.mkdir(parents=True, exist_ok=True)
    
    print('📋 Loading test cases from dataURI.txt...')
    test_cases = load_test_cases_from_datauri(DATAURI_FILE, n_cases=50)
    
    if not test_cases:
        print("❌ No test cases found in dataURI.txt. Exiting.")
        return
    
    print(f"✅ Found {len(test_cases)} test cases")
    
    print(f'\n📂 Processing images...')
    print(f'   Source: {DESIGN2CODE_DIR}')
    print(f'   Clean images: {CLEAN_IMGS_DIR}')
    print(f'   Perturbed images: {PERTURBED_IMGS_DIR}')
    
    success_count = 0
    failed_count = 0
    
    for test_id in test_cases:
        png_file = DESIGN2CODE_DIR / f"{test_id}.png"
        
        if not png_file.exists():
            print(f"   ⚠️  Warning: {test_id}.png not found in Design2Code/")
            failed_count += 1
            continue
        
        try:
            # 1. Copy to clean_imgs
            clean_img_path = CLEAN_IMGS_DIR / f"{test_id}.png"
            shutil.copy2(png_file, clean_img_path)
            
            # 2. Load image and apply perturbation
            img = Image.open(clean_img_path)
            perturbed_img = perturb_image(img, strength=0.05)
            
            # 3. Save perturbed image
            perturbed_img_path = PERTURBED_IMGS_DIR / f"{test_id}.png"
            perturbed_img.save(perturbed_img_path)
            
            success_count += 1
            if success_count % 10 == 0:
                print(f"   ✅ Processed {success_count} images...")
                
        except Exception as e:
            print(f"   ❌ Error processing {test_id}.png: {e}")
            failed_count += 1
    
    print(f'\n✅ Processing complete!')
    print(f'   Success: {success_count}')
    print(f'   Failed: {failed_count}')
    print(f'   Clean images saved to: {CLEAN_IMGS_DIR}')
    print(f'   Perturbed images saved to: {PERTURBED_IMGS_DIR}')


if __name__ == "__main__":
    main()
