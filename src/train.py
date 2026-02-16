import numpy as np
from pathlib import Path
import joblib
import random
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report
from tqdm import tqdm
import xgboost as xgb

from src.dataset import build_training_dataset, load_dataset, save_dataset, add_hard_negatives
from src.preprocessing import preprocess_image
from src.features import count_hog_features, extract_all_features
from src.utils import read_yolo_labels, conversion_yolo_to_pixels, compute_iou
from src.detection import generate_proposals_selective_search
from src.constants import CLASS_NAMES, NUM_CLASSES, BACKGROUND_CLASS


def train_xgboost(X, y, variance=0.95):
    """
    Train XGBoost classifier.
    
    Args:
        X: features
        y: labels
        variance: variance to explain
    """    
    n_hog = count_hog_features()
    hog_part, other_part = X[:, :n_hog], X[:, n_hog:]

    hog_scaler = StandardScaler()
    hog_scaled = hog_scaler.fit_transform(hog_part)
    
    pca = PCA(n_components=variance)
    hog_reduced = pca.fit_transform(hog_scaled)
    X_trans = np.concatenate([hog_reduced, other_part], axis=1)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_trans)

    clf = xgb.XGBClassifier(
        eval_metric='mlogloss', 
        num_class=NUM_CLASSES, 
        objective='multi:softprob'
    )
    clf.fit(X_scaled, y)

    return clf, scaler, pca, hog_scaler

def save_classifier(classifier, output_dir):
    """Save classifier.
    
    Args:
        classifier: clf, scaler, pca, hog_scaler, (optional) sift_vocab
        output_dir: directory to save models
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if len(classifier) == 5:
        clf, scaler, pca, hog_scaler, sift_vocab = classifier
        joblib.dump(sift_vocab, output_dir / "sift_vocab.joblib")
    else:
        clf, scaler, pca, hog_scaler = classifier
    
    joblib.dump(clf, output_dir / "model.joblib")
    joblib.dump(scaler, output_dir / "scaler.joblib")
    joblib.dump(pca, output_dir / "pca.joblib")
    joblib.dump(hog_scaler, output_dir / "hog_scaler.joblib")
    print(f"classifier saved to {output_dir}")

def load_classifier(models_dir):
    """
    Load trained classifier.
    
    Args:
        models_dir: directory with saved models
    """
    models_dir = Path(models_dir)
    clf = joblib.load(models_dir / "model.joblib")
    scaler = joblib.load(models_dir / "scaler.joblib")
    pca = joblib.load(models_dir / "pca.joblib")
    hog_scaler = joblib.load(models_dir / "hog_scaler.joblib")
    
    # load SIFT vocabulary if exists
    sift_vocab_path = models_dir / "sift_vocab.joblib"
    if sift_vocab_path.exists():
        sift_vocab = joblib.load(sift_vocab_path)
        return clf, scaler, pca, hog_scaler, sift_vocab
    
    return clf, scaler, pca, hog_scaler

def train_pipeline(
        images_dir, 
        labels_dir, 
        models_dir, 
        dataset_dir=None, 
        dataset_exist=False, 
        n_samples=1, 
        max_negatives=None, 
        use_sift=False
    ):
    """
    Training pipeline.
    
    Args:
        images_dir: directory with images
        labels_dir: directory with labels
        models_dir: directory to save trained models
        dataset_dir: directory to save or load dataset
        dataset_exist: whether dataset already exists
        use_sift: whether to use SIFT BoVW features
    """
    sift_vocab = None
    
    if dataset_exist and dataset_dir:
        X, y = load_dataset(dataset_dir)
        # try to load existing SIFT vocabulary
        if use_sift:
            vocab_path = Path(models_dir) / "sift_vocab.joblib"
            if vocab_path.exists():
                sift_vocab = joblib.load(vocab_path)
                print("loaded existing SIFT vocabulary")
    else:
        result = build_training_dataset(
            images_dir, 
            labels_dir, 
            n_samples=n_samples, 
            max_negatives=max_negatives, 
            build_sift_vocab=use_sift
        )
        if use_sift:
            X, y, sift_vocab = result
        else:
            X, y = result
        if dataset_dir:
            save_dataset(X, y, dataset_dir)
    
    print(f"training on {len(y)} samples...")
    classifier = train_xgboost(X, y)
    
    if use_sift and sift_vocab is not None:
        clf, scaler, pca, hog_scaler = classifier
        classifier = (clf, scaler, pca, hog_scaler, sift_vocab)
    
    save_classifier(classifier, models_dir)
    return classifier


def hard_negative_mining(
        images_dir, 
        labels_dir, 
        classifier, 
        max_images=100, 
        max_per_image_nonbg=None, 
        max_total_bg=None,
        iou_threshold=0.3
):
    """Do hard negative mining.
    
    Args:
        images_dir: directory with images
        labels_dir: directory with labels
        classifier: trained classifier to use for mining
        max_images: maximum images to use for mining
        max_per_image_nonbg: maximum non-background hard negatives per image
        max_total_bg: maximum total background hard negatives across all images
        iou_threshold: IoU threshold to consider a detection as true positive
    """
    # handle classifier with or without SIFT vocabulary
    if len(classifier) == 5:
        clf, scaler, pca, hog_scaler, sift_vocab = classifier
    else:
        clf, scaler, pca, hog_scaler = classifier
        sift_vocab = None
    
    hard_features, hard_labels = [], []
    image_files = sorted(Path(images_dir).glob("*.jpg"))    
    if max_images is not None and max_images > 0 and max_images < len(image_files):
        image_files = random.sample(image_files, max_images)

    # counter to limit hard negatives
    total_bg_count = 0

    for img_path in tqdm(image_files, total=len(image_files)):
        
        label_path = Path(labels_dir) / f"{img_path.stem}.txt"
        image = preprocess_image(str(img_path))
        h, w = image['grayscale'].shape[:2]

        gt_boxes = []
        if label_path.exists():
            boxes = read_yolo_labels(str(label_path))
            gt_pixel = conversion_yolo_to_pixels(boxes, w, h)
            gt_boxes = [(int(b[0]), b[1], b[2], b[3], b[4]) for b in gt_pixel]

        # generate proposals from the same detection pipeline used in inference
        proposals = generate_proposals_selective_search(image['grayscale'])

        if not proposals:
            continue

        feats, props = [], []
        for (x1, y1, x2, y2) in proposals:
            if x2 <= x1 or y2 <= y1:
                continue
            feat = extract_all_features(image, (x1, y1, x2, y2), sift_vocabulary=sift_vocab)
            if feat is not None:
                feats.append(feat)
                props.append((x1, y1, x2, y2))

        if not feats:
            continue

        # ensure we don't add too many hard negatives per image
        per_image_nonbg_count = 0

        X = np.vstack(feats)
        n_hog = count_hog_features()
        hog_part = X[:, :n_hog]
        hog_scaled = hog_scaler.transform(hog_part)
        hog_reduced = pca.transform(hog_scaled)
        X_trans = np.concatenate([hog_reduced, X[:, n_hog:]], axis=1)
        X_scaled = scaler.transform(X_trans)
        preds = clf.predict(X_scaled)

        for i, (x1, y1, x2, y2) in enumerate(props):
            pred_cls = int(preds[i])

            # we only care about false positives (proposals predicted as non-background)
            if pred_cls == BACKGROUND_CLASS:
                continue
            
            is_fp = True
            for gt_cls, gx1, gy1, gx2, gy2 in gt_boxes:
                iou = compute_iou([x1, y1, x2, y2], [gx1, gy1, gx2, gy2])
                # if this proposal has high iou with a gt box of the same class, it's not a hard negative
                if iou > iou_threshold and gt_cls == pred_cls:
                    is_fp = False
                    break

            # if it's a hard negative, we assign it a label based on the best matching gt box 
            # or background if no good match, and add it to the hard negatives list
            if is_fp:
                best_iou, best_cls = 0, BACKGROUND_CLASS
                for gt_cls, gx1, gy1, gx2, gy2 in gt_boxes:
                    iou = compute_iou([x1, y1, x2, y2], [gx1, gy1, gx2, gy2])
                    if iou > best_iou:
                        best_iou, best_cls = iou, gt_cls
                assigned_label = best_cls if best_iou > iou_threshold else BACKGROUND_CLASS

                if assigned_label != BACKGROUND_CLASS:
                    if max_per_image_nonbg and per_image_nonbg_count >= max_per_image_nonbg:
                        continue
                    per_image_nonbg_count += 1
                else:
                    if max_total_bg and total_bg_count >= max_total_bg:
                        continue
                    total_bg_count += 1

                hard_features.append(feats[i])
                hard_labels.append(assigned_label)

    print(f"created {len(hard_features)} hard negatives")
    return hard_features, hard_labels

def retrain_with_hard_negatives(images_dir, labels_dir, models_dir, dataset_dir, max_images=1000, max_per_image_nonbg=None, max_total_bg=None):
    """
    Retrain classifier with hard negatives.
    
    Args:
        images_dir: directory with images
        labels_dir: directory with labels
        models_dir: directory with saved models
        dataset_dir: directory with existing dataset
        max_images: maximum images to use for hard negative mining
        max_per_image_nonbg: maximum non-background hard negatives per image
    """
    X, y = load_dataset(dataset_dir)
    classifier = load_classifier(models_dir)

    orig_unique, orig_counts = np.unique(y, return_counts=True)
    
    hard_feats, hard_labels = hard_negative_mining(images_dir, labels_dir, classifier, max_images=max_images, max_per_image_nonbg=max_per_image_nonbg, max_total_bg=max_total_bg)
    if hard_feats:
        X, y = add_hard_negatives(X, y, hard_feats, hard_labels)
        save_dataset(X, y, dataset_dir)
    
    print("distribution of classes in the original training set:")
    for cls, count in zip(orig_unique, orig_counts):
        print(f"  {CLASS_NAMES[cls]}: {count}")
    print("distribution of classes in the hard negatives that were added:")
    unique, counts = np.unique(hard_labels, return_counts=True)
    for cls, count in zip(unique, counts):
        print(f"  {CLASS_NAMES[cls]}: {count}")
    print("distribution of classes in the new training set:")
    unique, counts = np.unique(y, return_counts=True)
    for cls, count in zip(unique, counts):
        print(f"  {CLASS_NAMES[cls]}: {count}")
    
    print(f"retraining on {len(y)} samples...")
    if len(classifier) == 5:
        clf, scaler, pca, hog_scaler, sift_vocab = classifier
        classifier = train_xgboost(X, y)
        clf, scaler, pca, hog_scaler = classifier
        classifier = (clf, scaler, pca, hog_scaler, sift_vocab)
    else:
        classifier = train_xgboost(X, y)
    save_classifier(classifier, models_dir)

    return classifier

def validate_pipeline(models_dir, images_dir, labels_dir, nb_samples=1, max_negatives=None, use_sift=False, sift_vocab=None):
    """
    Validation for a trained classifier.
    
    Args:
        models_dir: directory with saved models
        images_dir: directory with validation images
        labels_dir: directory with validation labels
        nb_samples: number of samples to use for validation
        max_negatives: maximum negative samples to use for validation
        use_sift: whether to use SIFT features
        sift_vocab: SIFT vocabulary to use for feature extraction
    """
    classifier = load_classifier(models_dir)

    if use_sift:
        if len(classifier) == 5:
            clf, scaler, pca, hog_scaler, sift_vocab = classifier
        else:
            print("SIFT vocabulary not found in the loaded classifier.")
            clf, scaler, pca, hog_scaler = classifier
    else:
        clf, scaler, pca, hog_scaler = classifier
    
    X, y = build_training_dataset(images_dir, labels_dir, n_samples=nb_samples, max_negatives=max_negatives, sift_vocab=sift_vocab)

    n_hog = count_hog_features()
    hog_part = X[:, :n_hog]
    hog_scaled = hog_scaler.transform(hog_part)
    hog_reduced = pca.transform(hog_scaled)
    X_trans = np.concatenate([hog_reduced, X[:, n_hog:]], axis=1)
    X_scaled = scaler.transform(X_trans)
    
    y_pred = clf.predict(X_scaled)
    print(classification_report(y, y_pred, target_names=[CLASS_NAMES[i] for i in range(NUM_CLASSES)]))
