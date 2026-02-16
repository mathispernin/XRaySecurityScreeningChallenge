import glob
import numpy as np
import os
import random
from pathlib import Path
from tqdm import tqdm

from src.preprocessing import preprocess_image
from src.features import extract_all_features, extract_sift_descriptors
from src.utils import read_yolo_labels, conversion_yolo_to_pixels, compute_iou
from src.constants import CLASS_NAMES, NUM_CLASSES, BACKGROUND_CLASS, SCALES


def generate_samples(
        images_dir, 
        labels_dir, 
        n_samples=1, 
        scales=SCALES, 
        max_negatives=None, 
        collect_sift=False, 
        sift_vocabulary=None
    ):
    """
    Generate samples.

    Args:
        images_dir: directory with images
        labels_dir: directory with labels
        n_samples: number of negative samples per image
        scales: list of scales for negative samples
        max_negatives: maximum total negative samples to create across all images
        collect_sift: if True, collect SIFT descriptors for vocabulary building
        sift_vocabulary: SIFT vocabulary for BoVW feature extraction
    """
    samples = {c: [] for c in range(NUM_CLASSES)}
    sift_descriptors = [] if collect_sift else None
    image_files = sorted(glob.glob(f"{images_dir}/*.jpg"))

    # we track total nb of negatives across all images
    total_negatives = 0

    for img_path in tqdm(image_files, total=len(image_files)):
        label_path = os.path.join(labels_dir, os.path.splitext(os.path.basename(img_path))[0] + ".txt")

        image = preprocess_image(img_path)
        image_gray = image['grayscale']

        image_h, image_w = image_gray.shape[:2]
        boxes = read_yolo_labels(label_path)
        if not boxes:
            print(f"no boxes found in {label_path}")
            continue
        pixel_bboxes = conversion_yolo_to_pixels(boxes, image_w, image_h)
        gt_boxes = [[b[1], b[2], b[3], b[4]] for b in pixel_bboxes]

        # add bbox samples
        for bbox in pixel_bboxes:
            cls = int(bbox[0])
            x1, y1, x2, y2 = int(bbox[1]), int(bbox[2]), int(bbox[3]), int(bbox[4])

            # extract features for this bbox
            features = extract_all_features(image, (x1, y1, x2, y2), crop_with_padding=True, sift_vocabulary=sift_vocabulary)
            if features is not None:
                samples[cls].append(features)
            
            # collect SIFT descriptors if requested
            if collect_sift:
                crop = image['grayscale'][y1:y2, x1:x2]
                desc = extract_sift_descriptors(crop)
                if desc is not None:
                    sift_descriptors.append(desc)

        if max_negatives is not None and total_negatives >= max_negatives:
            continue

        # add negative samples
        # we sample random windows and check that they have low IoU with all gt boxes.
        # we ensure diversity by sampling from different images 
        attempts, saved = 0, 0
        while saved < n_samples and attempts < n_samples * 20:
            attempts += 1

            # sample window size from log-normal distribution
            # parameters are chosen to reproduce the typical object sizes in the training set
            win_w = int(np.random.lognormal(mean=5.0, sigma=0.55))
            ratio = np.random.lognormal(mean=0.2, sigma=0.6)
            win_h = int(win_w / ratio)
            win_w = int(np.clip(win_w, 30, 500))
            win_h = int(np.clip(win_h, 30, 400))

            scale = random.choice(scales)
            win_w_s, win_h_s = int(win_w * scale), int(win_h * scale)

            if win_w_s >= image_w or win_h_s >= image_h:
                continue
            
            x1, y1 = random.randint(0, image_w - win_w_s), random.randint(0, image_h - win_h_s)
            x2, y2 = x1 + win_w_s, y1 + win_h_s

            iou_thresh = 0.1 # low iou threshold to ensure negative samples are not too close to any gt box
            if all(compute_iou([x1, y1, x2, y2], gt) <= iou_thresh for gt in gt_boxes):
                features = extract_all_features(image, (x1, y1, x2, y2), sift_vocabulary=sift_vocabulary)
                samples[BACKGROUND_CLASS].append(features)
                saved += 1
                total_negatives += 1
                
                # collect SIFT descriptors if requested
                if collect_sift:
                    crop = image['grayscale'][y1:y2, x1:x2]
                    desc = extract_sift_descriptors(crop)
                    if desc is not None:
                        sift_descriptors.append(desc)

                if max_negatives is not None and total_negatives >= max_negatives:
                    break

    if collect_sift:
        return samples, sift_descriptors
    return samples

def build_training_dataset(
        images_dir, 
        labels_dir, 
        n_samples=1, 
        scales=SCALES, 
        max_negatives=None, 
        build_sift_vocab=False, 
        sift_vocab=None
    ):
    """
    Build training dataset.

    Args:
        images_dir: directory with images
        labels_dir: directory with labels
        n_samples: number of negative samples per image
        scales: list of scales for negative samples
        max_negatives: maximum total negative samples to create across all images
        build_sift_vocab: if True, build SIFT vocabulary before extracting features
        sift_vocab: built SIFT vocabulary for BoVW feature extraction
    """    
    # pass 1: collect SIFT descriptors if needed
    if build_sift_vocab:
        from src.features import build_sift_vocabulary
        print("pass 1/2: collecting SIFT descriptors for vocabulary...")
        _, sift_descriptors = generate_samples(
            images_dir,
            labels_dir,
            n_samples=n_samples,
            scales=scales,
            max_negatives=max_negatives,
            collect_sift=True,
        )
        print(f"collected {len(sift_descriptors)} SIFT descriptors")
        sift_vocab = build_sift_vocabulary(sift_descriptors)
    
    # pass 2: extract features
    print(f"pass {2 if build_sift_vocab else 1}/{2 if build_sift_vocab else 1}: extracting features...")
    samples = generate_samples(
        images_dir,
        labels_dir,
        n_samples=n_samples,
        scales=scales,
        max_negatives=max_negatives,
        collect_sift=False,
        sift_vocabulary=sift_vocab,
    )

    X_list, y_list = [], []
    for cls, features in samples.items():
        if features:
            X_list.append(np.array(features))
            y_list.append(np.full(len(features), cls))
            print(f"class {cls} ({CLASS_NAMES[cls]}): {len(features)} samples")

    X = np.vstack(X_list)
    y = np.concatenate(y_list)
    idx = np.random.permutation(len(y))
    
    if build_sift_vocab:
        return X[idx], y[idx], sift_vocab
    return X[idx], y[idx]

def save_dataset(X, y, output_dir, suffix=""):
    """Save dataset."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / f"X{suffix}.npy", X)
    np.save(output_dir / f"y{suffix}.npy", y)
    print(f"dataset saved to {output_dir}")
    return

def load_dataset(input_dir, suffix=""):
    """Load dataset."""
    input_dir = Path(input_dir)
    X = np.load(input_dir / f"X{suffix}.npy")
    y = np.load(input_dir / f"y{suffix}.npy")
    print(f"dataset loaded from {input_dir}, {len(y)} samples")
    return X, y

def add_hard_negatives(X, y, hard_negatives, hard_labels):
    """Add samples to an existing dataset."""
    X_new = np.vstack([X, np.array(hard_negatives)])
    y_new = np.concatenate([y, np.array(hard_labels)])
    idx = np.random.permutation(len(y_new))
    return X_new[idx], y_new[idx]
