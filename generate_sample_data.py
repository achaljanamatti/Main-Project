import os
import numpy as np
import cv2
import logging
from sklearn.datasets import make_classification

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
OUTPUT_DIR = "dataset"
NUM_SAMPLES = 200  # 100 of each type
IMAGE_SIZE = (128, 128)

def generate_synthetic_bone_marrow_image(is_leukemia=False):
    """
    Generate a synthetic bone marrow sample image
    
    Args:
        is_leukemia (bool): Whether to generate a leukemic or normal cell image
        
    Returns:
        numpy.ndarray: A synthetic bone marrow cell image
    """
    # Create a base image with appropriate color for bone marrow background
    img = np.ones((IMAGE_SIZE[0], IMAGE_SIZE[1], 3), dtype=np.uint8) * 240  # Light gray background
    
    # Add some texture to the background
    noise = np.random.randint(0, 15, (IMAGE_SIZE[0], IMAGE_SIZE[1], 3), dtype=np.uint8)
    img = cv2.add(img, noise)
    
    # Draw cell nucleus
    center_x = np.random.randint(40, IMAGE_SIZE[0] - 40)
    center_y = np.random.randint(40, IMAGE_SIZE[1] - 40)
    
    if is_leukemia:
        # Leukemic cells tend to have larger, irregular nuclei with uneven chromatin
        radius = np.random.randint(25, 35)
        nucleus_color = (80, 50, 130)  # Purple-ish for leukemic nuclei
        
        # Draw irregular shape
        points = []
        for i in range(8):
            angle = 2 * np.pi * i / 8
            r = radius * (0.8 + 0.4 * np.random.random())
            x = int(center_x + r * np.cos(angle))
            y = int(center_y + r * np.sin(angle))
            points.append((x, y))
        
        points = np.array(points, dtype=np.int32)
        cv2.fillPoly(img, [points], nucleus_color)
        
        # Add chromatin pattern inside nucleus (characteristic of leukemic cells)
        for _ in range(15):
            cx = center_x + np.random.randint(-radius//2, radius//2)
            cy = center_y + np.random.randint(-radius//2, radius//2)
            cr = np.random.randint(2, 5)
            cv2.circle(img, (cx, cy), cr, (40, 20, 80), -1)
    else:
        # Normal bone marrow cells have smaller, regular nuclei
        radius = np.random.randint(15, 22)
        nucleus_color = (100, 100, 160)  # Blue-ish for normal nuclei
        
        # Draw regular circular nucleus
        cv2.circle(img, (center_x, center_y), radius, nucleus_color, -1)
        
        # Add some texture to the nucleus
        for _ in range(5):
            cx = center_x + np.random.randint(-radius//2, radius//2)
            cy = center_y + np.random.randint(-radius//2, radius//2)
            cr = np.random.randint(1, 3)
            cv2.circle(img, (cx, cy), cr, (80, 80, 140), -1)
    
    # Draw cell cytoplasm
    cytoplasm_radius = radius + np.random.randint(5, 12)
    cytoplasm_color = (210, 180, 180)  # Light reddish for cytoplasm
    cv2.circle(img, (center_x, center_y), cytoplasm_radius, cytoplasm_color, -1)
    
    # Draw nucleus on top of cytoplasm
    if is_leukemia:
        cv2.fillPoly(img, [points], nucleus_color)
    else:
        cv2.circle(img, (center_x, center_y), radius, nucleus_color, -1)
    
    # Add blur to make it more realistic
    img = cv2.GaussianBlur(img, (3, 3), 0)
    
    return img

def main():
    """
    Generate synthetic bone marrow images for model training
    """
    # Create output directories
    os.makedirs(os.path.join(OUTPUT_DIR, "leukemia"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "non_leukemia"), exist_ok=True)
    
    logger.info(f"Generating {NUM_SAMPLES} synthetic bone marrow images...")
    
    # Generate leukemia and non-leukemia samples
    for i in range(NUM_SAMPLES // 2):
        # Generate leukemia sample
        img_leukemia = generate_synthetic_bone_marrow_image(is_leukemia=True)
        leukemia_path = os.path.join(OUTPUT_DIR, "leukemia", f"leukemia_{i:03d}.png")
        cv2.imwrite(leukemia_path, img_leukemia)
        
        # Generate non-leukemia sample
        img_normal = generate_synthetic_bone_marrow_image(is_leukemia=False)
        normal_path = os.path.join(OUTPUT_DIR, "non_leukemia", f"normal_{i:03d}.png")
        cv2.imwrite(normal_path, img_normal)
    
    logger.info(f"Generated {NUM_SAMPLES // 2} leukemia images and {NUM_SAMPLES // 2} normal images")
    logger.info(f"Images saved to {OUTPUT_DIR}/leukemia and {OUTPUT_DIR}/non_leukemia")

if __name__ == "__main__":
    main()