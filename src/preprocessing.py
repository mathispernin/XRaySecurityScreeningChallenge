import cv2
import numpy as np

from src.constants import BILATERAL_D, BILATERAL_SIGMA_COLOR, BILATERAL_SIGMA_SPACE, CLAHE_CLIP_LIMIT, CLAHE_TILE_GRID_SIZE


def load_image_raw(image_path):
    """Load raw image without preprocessing."""
    return cv2.imread(image_path)

def load_image_gray(image_path):
    """Load image as grayscale."""
    return cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

def load_image(image_source):
    """Loads image from path or validates numpy array."""
    if isinstance(image_source, str):
        image = cv2.imread(image_source)
    elif isinstance(image_source, np.ndarray):
        image = image_source
    else:
        raise TypeError("Input must be an image path (str) or a numpy array.")
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image

def preprocess_image(image_path):
    """
    Preprocessing pipeline with 2 branches:
        1. Material (HSV) -> For Color Histograms
        2. Texture (Enhanced Grayscale) -> For HOG, LBP, GLCM
    
    Args:
        image_source: Path to image or numpy array.
        
    Returns:
        Dictionary containing processed image versions.
    """
    img_bgr = load_image(image_path)

    # branch 1: material (HSV) ; light filter to remove noise but keep color transitions sharp
    img_clean = cv2.bilateralFilter(img_bgr, BILATERAL_D, BILATERAL_SIGMA_COLOR, BILATERAL_SIGMA_SPACE)
    img_hsv = cv2.cvtColor(img_clean, cv2.COLOR_BGR2HSV)

    # branch 2: texture (enhanced grayscale)
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    # CLAHE to enhance local contrast
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_GRID_SIZE)
    img_enhanced_gray = clahe.apply(img_gray)
    
    # second bilateral pass on gray to remove noise while preserving edges
    img_enhanced_gray = cv2.bilateralFilter(img_enhanced_gray, BILATERAL_D, 50, 50)

    return {
        "original_bgr": img_bgr,
        "hsv": img_hsv,
        "grayscale": img_enhanced_gray,
    }
