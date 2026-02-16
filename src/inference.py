import pandas as pd
from pathlib import Path
from tqdm import tqdm

from src.preprocessing import preprocess_image
from src.detection import detect_objects
from src.train import load_classifier


def run_inference(test_dir, classifier, threshold=0.5, **kwargs):
    """Run inference on test set."""
    test_dir = Path(test_dir)
    image_files = sorted(test_dir.glob("*.png")) + sorted(test_dir.glob("*.jpg"))
    
    predictions = {}
    for img_path in tqdm(image_files, desc="inference"):
        frame_id = img_path.stem
        image = preprocess_image(str(img_path))
        detections = detect_objects(image, classifier, threshold=threshold, **kwargs)
        predictions[frame_id] = detections
    
    return predictions

def predictions_to_submission(predictions, output_csv):
    """Convert predictions to submission CSV."""
    data = []
    for frame_id in sorted(predictions.keys()):
        boxes = [[int(b[0]), float(b[1]), float(b[2]), float(b[3]), float(b[4])] for b in predictions[frame_id]]
        data.append({'frame_id': frame_id, 'bbs': str(boxes)})
    
    pd.DataFrame(data).to_csv(output_csv, index=False)
    print(f"submission saved to {output_csv}")

def generate_submission(test_dir, models_dir, output_csv, threshold=0.5, **kwargs):
    """Complete inference pipeline."""
    print("loading classifier...")
    classifier = load_classifier(models_dir)
    
    print("running inference...")
    predictions = run_inference(test_dir, classifier, threshold=threshold, **kwargs)
    predictions_to_submission(predictions, output_csv)
    print("inference completed.")
