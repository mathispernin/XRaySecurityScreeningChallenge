import cv2
import numpy as np

from src.preprocessing import preprocess_image
from src.features import count_hog_features, extract_all_features
from src.constants import BACKGROUND_CLASS


def apply_nms(proposals, iou_threshold=0.5):
    """Apply Non-Maximum Suppression (NMS) to filter overlapping proposals."""
    if len(proposals) == 0:
        return []

    # format to [x, y, w, h] for openCV
    boxes_wh = []
    for (x1, y1, x2, y2) in proposals:
        boxes_wh.append([x1, y1, x2 - x1, y2 - y1])

    # we use dummy scores since we just want to filter by iou
    scores = [1.0] * len(boxes_wh)
    indices = cv2.dnn.NMSBoxes(boxes_wh, scores, score_threshold=0.5, nms_threshold=iou_threshold)

    filtered_proposals = []
    if len(indices) > 0:
        for i in indices.flatten():
            filtered_proposals.append(proposals[i])

    return filtered_proposals

def generate_proposals_selective_search(image, min_area=3000, max_area_ratio=0.6, top_k=1000, iou_thresh=0.7):
    """Generate region proposals using Selective Search + NMS."""
    h, w = image.shape[:2]
    max_area = h * w * max_area_ratio

    if len(image.shape) == 2:
        img_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        img_color = image

    # generate proposals using Selective Search
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(img_color)
    ss.switchToSelectiveSearchFast()
    rects = ss.process()
    proposals = []
    
    # filter proposals by size and aspect ratio, and keep top_k
    for i, rect in enumerate(rects):
        if i >= top_k: 
            break
            
        x, y, bw, bh = rect
        area = bw * bh
        
        if min_area < area < max_area:
            aspect_ratio = bw / float(bh)
            if 0.1 < aspect_ratio < 10.0:
                proposals.append((x, y, x + bw, y + bh))
    
    # apply NMS to filter overlapping proposals
    final_proposals = apply_nms(proposals, iou_threshold=iou_thresh)
    final_proposals = proposals
    
    return final_proposals

def global_nms(detections, iou_threshold=0.4):
    """
    Apply global NMS across all classes.
    
    Args:
        detections: list of detections in format [class, x1, y1, x2, y2, score]
        iou_threshold: iou threshold for global NMS
    """
    if not detections:
        return []
    
    boxes = np.array([[d[1], d[2], d[3], d[4]] for d in detections])
    scores = np.array([d[5] for d in detections])

    # convert to [x, y, w, h] for openCV
    boxes_xywh = [[b[0], b[1], b[2] - b[0], b[3] - b[1]] for b in boxes]

    indices = cv2.dnn.NMSBoxes(boxes_xywh, scores.tolist(), 0.0, iou_threshold)

    if len(indices) == 0:
        return []
    
    indices = [int(i[0]) if hasattr(i, "__len__") else int(i) for i in indices]
    return [detections[i] for i in indices]

def detect_objects(image, classifier, threshold=0.5, nms_threshold=0.4, max_detections=1):
    """
    Detect objects using proposal generator above and XGBoost multiclass classifier.
    
    Args:
        image: dict with keys 'grayscale', 'hsv' (preprocessed image)
        classifier: (clf, scaler, pca, hog_scaler) or (clf, scaler, pca, hog_scaler, sift_vocab)
        threshold: confidence threshold
        nms_threshold: NMS iou threshold
        max_detections: maximum detections to return
    """
    # handle classifier with or without SIFT vocabulary
    if len(classifier) == 5:
        clf, scaler, pca, hog_scaler, sift_vocab = classifier
    else:
        clf, scaler, pca, hog_scaler = classifier
        sift_vocab = None
    
    h_orig, w_orig = image['grayscale'].shape[:2]
    
    proposals = generate_proposals_selective_search(image['grayscale'])

    if not proposals:
        return []
    
    features_list = []
    valid_proposals = []
    for (x1, y1, x2, y2) in proposals:
        if x2 <= x1 or y2 <= y1:
            continue
        feat = extract_all_features(image, (x1, y1, x2, y2), sift_vocabulary=sift_vocab)
        if feat is not None:
            features_list.append(feat)
            valid_proposals.append((x1, y1, x2, y2))
    
    if not features_list:
        return []
    
    X = np.vstack(features_list)
    n_hog = count_hog_features()
    hog_part = X[:, :n_hog]
    hog_scaled = hog_scaler.transform(hog_part)
    hog_reduced = pca.transform(hog_scaled)
    X_trans = np.concatenate([hog_reduced, X[:, n_hog:]], axis=1)
    X_scaled = scaler.transform(X_trans)
    
    probas = clf.predict_proba(X_scaled)
    predictions = clf.predict(X_scaled)
    
    detections = []
    for i, (x1, y1, x2, y2) in enumerate(valid_proposals):
        pred_class = int(predictions[i])
        if pred_class == BACKGROUND_CLASS:
            continue
        
        conf = probas[i, pred_class]
        if conf >= threshold:
            detections.append((pred_class, x1, y1, x2, y2, conf))
    
    # global NMS to filter overlapping detections across all classes
    detections = global_nms(detections, nms_threshold)
    
    # sort by confidence and keep top max_detections
    detections.sort(key=lambda x: x[5], reverse=True)
    detections = detections[:max_detections]

    results = []
    for (cls, x1, y1, x2, y2, _) in detections:
        cx = np.clip(((x1 + x2) / 2) / w_orig, 0, 1)
        cy = np.clip(((y1 + y2) / 2) / h_orig, 0, 1)
        w = np.clip((x2 - x1) / w_orig, 0, 1)
        h = np.clip((y2 - y1) / h_orig, 0, 1)
        results.append([cls, cx, cy, w, h])
    
    return results

def detect_objects_in_image(image_path, classifier, **kwargs):
    """Detect objects from an image file."""
    image = preprocess_image(image_path)
    return detect_objects(image, classifier, **kwargs)
