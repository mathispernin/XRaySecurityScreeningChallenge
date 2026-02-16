import numpy as np
from pathlib import Path
from tqdm import tqdm

from src.preprocessing import preprocess_image
from src.detection import detect_objects
from src.utils import read_yolo_labels, compute_iou_yolo


def compute_frame_score(gt_boxes, pred_boxes, image_width=1, image_height=1):
    """
    Compute frame score.
    
    Args:
        gt_boxes: list of GT boxes [class_id, cx, cy, w, h]
        pred_boxes: list of predicted boxes [class_id, cx, cy, w, h]
        image_width: image width for iou computation
        image_height: image height for iou computation
    """
    if len(gt_boxes) == 0 and len(pred_boxes) == 0:
        return 1.0
    if len(gt_boxes) == 0 or len(pred_boxes) == 0:
        return 0.0
    
    matched_preds = set()
    total_iou = 0.0
    
    for gt in gt_boxes:
        gt_class = int(gt[0])
        best_iou = 0.0
        best_pred_idx = -1
        
        for pred_idx, pred in enumerate(pred_boxes):
            if pred_idx in matched_preds:
                continue
            
            pred_class = int(pred[0])
            if pred_class != gt_class:
                continue
            
            iou = compute_iou_yolo(gt, pred, image_width, image_height)
            if iou > best_iou:
                best_iou = iou
                best_pred_idx = pred_idx
        
        if best_pred_idx >= 0:
            matched_preds.add(best_pred_idx)
            total_iou += best_iou
    
    denominator = max(len(gt_boxes), len(pred_boxes))
    return total_iou / denominator

def evaluate_on_dataset(images_dir, labels_dir, classifier, threshold=0.5, nb_of_images=None, **detection_kwargs):
    """
    Evaluate detector on a dataset.
    
    Args:
        images_dir: directory with images
        labels_dir: directory with labels
        classifier: trained classifier
        threshold: detection threshold per class
        nb_of_images: limits evaluation to this many images
        **detection_kwargs: additional arguments for detection
        
    Returns:
        mean score across all frames
    """
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)
    image_files = sorted(images_dir.glob("*.jpg"))

    if nb_of_images is not None:
        image_files = image_files[:nb_of_images]
        
    scores = []

    for img_path in tqdm(image_files, total=len(image_files)):
        label_path = labels_dir / f"{img_path.stem}.txt"
        gt_boxes = read_yolo_labels(str(label_path)) if label_path.exists() else []
        image = preprocess_image(str(img_path))
        pred_boxes = detect_objects(image, classifier, threshold=threshold, **detection_kwargs)
        scores.append(compute_frame_score(gt_boxes, pred_boxes))
    
    return np.mean(scores) if scores else 0.0
