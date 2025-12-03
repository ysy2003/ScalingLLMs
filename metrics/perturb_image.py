# perturbations.py
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
import random

def perturb_image(img: Image.Image, strength: float = 0.05) -> Image.Image:
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
