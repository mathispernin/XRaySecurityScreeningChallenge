import cv2
import numpy as np
import random
from skimage.feature import hog, local_binary_pattern, graycomatrix, graycoprops
from sklearn.cluster import MiniBatchKMeans

from src.constants import HOG_WINDOW_SIZE, HOG_ORIENTATIONS, HOG_PIXELS_PER_CELL, HOG_CELLS_PER_BLOCK, HOG_BLOCK_NORM
from src.constants import LBP_P, LBP_R, LBP_METHOD, LBP_NUM_BINS
from src.constants import HIST_HUE_BINS, HIST_SAT_BINS, GLCM_DISTANCES, GLCM_ANGLES, GLCM_LEVELS, GLCM_PROPS
from src.constants import MAX_TEXTURE_SIZE, SIFT_N_FEATURES, BOVW_VOCAB_SIZE


def resize_with_padding(image, target_size):
    """Resize image to target_size but maintains aspect ratio and padding with border replication."""
    target_w, target_h = target_size
    h, w = image.shape[:2]
    
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    
    resized = cv2.resize(image, (new_w, new_h))
    
    delta_w = target_w - new_w
    delta_h = target_h - new_h
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)
    
    return cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_REPLICATE)

def extract_crop_with_padding(image, x1, y1, x2, y2, offset_ratio=0.15, padding_ratio=None):
    """
    Extract a crop from an image.

    Args:
        image: source image
        x1, y1, x2, y2: bbox coordinates (in pixels)
        padding_ratio: if None, sampled uniformly in [0.01, 0.15]
        offset_ratio: max shift of bbox center
    """
    if padding_ratio is None:
        padding_ratio = random.uniform(0.01, 0.15)

    image_h, image_w = image.shape[:2]
    bbox_w = x2 - x1
    bbox_h = y2 - y1

    if offset_ratio and offset_ratio > 0:
        shift_x = random.uniform(-offset_ratio, offset_ratio) * bbox_w
        shift_y = random.uniform(-offset_ratio, offset_ratio) * bbox_h
    else:
        shift_x = 0
        shift_y = 0

    cx = (x1 + x2) / 2.0 + shift_x
    cy = (y1 + y2) / 2.0 + shift_y
    x1_shift = int(round(cx - bbox_w / 2.0))
    y1_shift = int(round(cy - bbox_h / 2.0))
    x2_shift = int(round(cx + bbox_w / 2.0))
    y2_shift = int(round(cy + bbox_h / 2.0))
    pad_x = int(bbox_w * padding_ratio)
    pad_y = int(bbox_h * padding_ratio)
    x1_pad = max(0, x1_shift - pad_x)
    y1_pad = max(0, y1_shift - pad_y)
    x2_pad = min(image_w, x2_shift + pad_x)
    y2_pad = min(image_h, y2_shift + pad_y)

    if x2_pad <= x1_pad or y2_pad <= y1_pad:
        return None
    return image[y1_pad:y2_pad, x1_pad:x2_pad]

def extract_hog_features(crop_gray):
    """Extract HOG features."""
    features = hog(
        crop_gray,
        orientations=HOG_ORIENTATIONS,
        pixels_per_cell=HOG_PIXELS_PER_CELL,
        cells_per_block=HOG_CELLS_PER_BLOCK,
        block_norm=HOG_BLOCK_NORM,
        visualize=False,
        feature_vector=True
    )
    return features

def extract_lbp_features(crop_gray):
    """Extract LBP features (texture)."""
    radius = LBP_R if isinstance(LBP_R, list) else [LBP_R]
    features = []
    
    for r in radius:
        lbp = local_binary_pattern(crop_gray, LBP_P, r, method=LBP_METHOD)
        hist, _ = np.histogram(lbp.ravel(), bins=LBP_NUM_BINS, range=(0, LBP_NUM_BINS))
        hist = hist.astype(np.float32) / (hist.sum() + 1e-6)
        features.append(hist)
    
    return np.concatenate(features)

def extract_color_features(crop_hsv):
    """Extract Color Histograms."""
    # Hue histogram
    hist_h = cv2.calcHist([crop_hsv], [0], None, [HIST_HUE_BINS], [0, 180])
    cv2.normalize(hist_h, hist_h)
    
    # saturation histogram
    hist_s = cv2.calcHist([crop_hsv], [1], None, [HIST_SAT_BINS], [0, 256])
    cv2.normalize(hist_s, hist_s)
    
    return np.concatenate([hist_h.flatten(), hist_s.flatten()])

def extract_glcm_features(crop_gray):
    """Extract Texture features using Gray Level Co-occurrence Matrix (GLCM)."""
    # quantize to GLCM_LEVELS
    quantized = (crop_gray / (256 / GLCM_LEVELS)).astype(np.uint8)
    glcm = graycomatrix(quantized, distances=GLCM_DISTANCES, angles=GLCM_ANGLES, levels=GLCM_LEVELS, symmetric=True, normed=True)
    
    features = []
    for prop in GLCM_PROPS:
        feat = graycoprops(glcm, prop).flatten()
        features.extend(feat)
    
    return np.array(features)

def extract_sift_descriptors(crop_gray):
    """Extract SIFT descriptors from a crop."""
    sift = cv2.SIFT_create(nfeatures=SIFT_N_FEATURES)
    keypoints, descriptors = sift.detectAndCompute(crop_gray, None)
    return descriptors

def extract_sift_bovw_features(crop_gray, vocabulary):
    """
    Extract BoVW histogram from SIFT descriptors.
    
    Args:
        crop_gray: grayscale crop
        vocabulary: KMeans model or cluster centers (shape: vocab_size x 128)
    """
    descriptors = extract_sift_descriptors(crop_gray)
    
    if descriptors is None or len(descriptors) == 0:
        vocab_size = vocabulary.n_clusters if hasattr(vocabulary, 'n_clusters') else len(vocabulary)
        return np.zeros(vocab_size, dtype=np.float32)
    
    # assign each descriptor to nearest visual word
    if hasattr(vocabulary, 'predict'):
        labels = vocabulary.predict(descriptors)
    else:
        # vocabulary is cluster centers array
        from scipy.spatial.distance import cdist
        distances = cdist(descriptors, vocabulary, metric='euclidean')
        labels = np.argmin(distances, axis=1)
    
    # build histogram
    vocab_size = vocabulary.n_clusters if hasattr(vocabulary, 'n_clusters') else len(vocabulary)
    hist, _ = np.histogram(labels, bins=vocab_size, range=(0, vocab_size))
    hist = hist.astype(np.float32)
    norm = np.linalg.norm(hist)
    if norm > 0:
        hist = hist / norm
    
    return hist

def build_sift_vocabulary(all_descriptors, vocab_size=BOVW_VOCAB_SIZE):
    """
    Build BoVW vocabulary from all SIFT descriptors using Mini-Batch K-Means.
    
    Args:
        all_descriptors: List of descriptor arrays or single concatenated array
        vocab_size: Number of visual words
    """
    if isinstance(all_descriptors, list):
        all_descriptors = np.vstack([d for d in all_descriptors if d is not None and len(d) > 0])
    
    print(f"building BoVW vocabulary from {len(all_descriptors)} descriptors...")
    kmeans = MiniBatchKMeans(n_clusters=vocab_size, batch_size=1000, random_state=42, verbose=0)
    kmeans.fit(all_descriptors)
    print(f"vocabulary built (with ){vocab_size} words)")
    
    return kmeans

def extract_all_features(preprocessed_data, box, window_size=HOG_WINDOW_SIZE, crop_with_padding=False, sift_vocabulary=None):
    """
    Main function to extract all features for a specific bbox.
    
    Args:
        preprocessed_data: Dictionary with grayscale and HSV images
        box: x1, y1, x2, y2 coordinates of the bbox (in pixels)
        window_size: target size for HOG input
        crop_with_padding: if True, apply padding to the crop before feature extraction
        sift_vocabulary: if provided, extract SIFT BoVW features using this vocabulary
    """
    x1, y1, x2, y2 = box
    h_img, w_img = preprocessed_data['grayscale'].shape

    if crop_with_padding:
        crop_gray = extract_crop_with_padding(preprocessed_data['grayscale'], x1, y1, x2, y2, padding_ratio=0.15)
        crop_hsv = extract_crop_with_padding(preprocessed_data['hsv'], x1, y1, x2, y2, padding_ratio=0.15)
    else:
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w_img - 1, x2), min(h_img - 1, y2)
        crop_gray = preprocessed_data['grayscale'][y1:y2, x1:x2]
        crop_hsv = preprocessed_data['hsv'][y1:y2, x1:x2]

    h_crop, w_crop = crop_gray.shape
    if h_crop > MAX_TEXTURE_SIZE or w_crop > MAX_TEXTURE_SIZE:
        scale = MAX_TEXTURE_SIZE / max(h_crop, w_crop)
        new_w, new_h = int(w_crop * scale), int(h_crop * scale)
        crop_texture_input = cv2.resize(crop_gray, (new_w, new_h), interpolation=cv2.INTER_AREA)
    else:
        crop_texture_input = crop_gray
    
    # HOG features
    crop_hog_input = resize_with_padding(crop_gray, window_size)
    ft_hog = extract_hog_features(crop_hog_input)
    
    # LBP features
    ft_lbp = extract_lbp_features(crop_texture_input)
    
    # GLCM features
    ft_glcm = extract_glcm_features(crop_texture_input)
    
    # Color features
    ft_color = extract_color_features(crop_hsv)

    # SIFT features (BoVW)
    if sift_vocabulary is not None:
        ft_sift = extract_sift_bovw_features(crop_gray, sift_vocabulary)
        return np.concatenate([ft_hog, ft_lbp, ft_glcm, ft_color, ft_sift])
    
    return np.concatenate([ft_hog, ft_lbp, ft_glcm, ft_color])

def count_hog_features():
    """Utility to count the number of HOG features based on current parameters."""
    dummy_image = np.zeros(HOG_WINDOW_SIZE, dtype=np.uint8)
    features = extract_hog_features(dummy_image)
    return len(features)
