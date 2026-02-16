import cv2
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path


def yolo_to_submission_csv(yolo_dir, output_csv):
    """
    Convert YOLO format .txt files to submission format CSV.

    Args:
        yolo_dir: Directory containing YOLO format .txt files (one per frame)
        output_csv: Path to output submission.csv
    """
    data = []

    for txt_file in sorted(Path(yolo_dir).glob("*.txt")):
        frame_id = txt_file.stem
        boxes = []

        with open(txt_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    cls, cx, cy, w, h = map(float, parts)
                    boxes.append([int(cls), cx, cy, w, h])

        data.append({'frame_id': frame_id, 'bbs': str(boxes)})

    pd.DataFrame(data).to_csv(output_csv, index=False)

def display_image(image, title=None):
    """Display an image"""
    plt.figure(figsize=(10, 10))
    if len(image.shape) == 2:
        plt.imshow(image, cmap='gray')
    else:
        plt.imshow(image)
    if title:
        plt.title(title)
    plt.axis('off')
    plt.show()

def read_yolo_labels(label_file):
    """Read YOLO format labels from a YOLO .txt label file."""
    boxes = []
    with open(label_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                cls, cx, cy, w, h = map(float, parts)
                boxes.append([int(cls), cx, cy, w, h])
    return boxes

def display_image_with_bboxes(image, bboxes=None, bbox_as_file=False):
    """
    Display an image with bounding boxes.

    Args:
        image: Path to the image file or image array.
        bboxes: List of bounding boxes in format [class, cx, cy, w, h] or path to bbox file.
        bbox_as_file: If True, bboxes are provided as a path to a txt file with bounding boxes.
            If False, bboxes are provided directly as a list.
    """
    if isinstance(image, str):
        image = cv2.imread(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if bbox_as_file and bboxes:
        boxes = read_yolo_labels(bboxes)
    elif bboxes:
        boxes = bboxes
    else:
        display_image(image)
        return

    height, width, _ = image.shape
    for box in boxes:
        _, cx, cy, w, h = box
        x1 = int((cx - w / 2) * width)
        y1 = int((cy - h / 2) * height)
        x2 = int((cx + w / 2) * width)
        y2 = int((cy + h / 2) * height)
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)

    display_image(image)

def conversion_yolo_to_pixels(bboxes, image_width, image_height):
    """
    Convert YOLO bboxes to pixel coordinates.

    Args:
        bboxes: List of bboxes in format [class, cx, cy, w, h]
        image_width: Width of the image (in pixels)
        image_height: Height of the image (in pixels)

    Returns:
        List of bounding boxes in pixel format [class, x1, y1, x2, y2]
    """
    pixel_bboxes = []
    for box in bboxes:
        cls, cx, cy, w, h = box
        x1 = int((cx - w / 2) * image_width)
        y1 = int((cy - h / 2) * image_height)
        x2 = int((cx + w / 2) * image_width)
        y2 = int((cy + h / 2) * image_height)
        pixel_bboxes.append([cls, x1, y1, x2, y2])
    return pixel_bboxes

def conversion_pixels_to_yolo(bboxes, image_width, image_height):
    """
    Convert pixel bboxes to YOLO format.

    Args:
        bboxes: List of bounding boxes in pixel format [class, x1, y1, x2, y2]
        image_width: Width of the image (in pixels)
        image_height: Height of the image (in pixels)
    """
    yolo_bboxes = []
    for box in bboxes:
        cls, x1, y1, x2, y2 = box
        cx = ((x1 + x2) / 2) / image_width
        cy = ((y1 + y2) / 2) / image_height
        w = (x2 - x1) / image_width
        h = (y2 - y1) / image_height
        yolo_bboxes.append([cls, cx, cy, w, h])
    return yolo_bboxes

def compute_iou(box1, box2):
    """Compute IoU between two boxes in pixel format [x1, y1, x2, y2]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area

    if union_area == 0:
        return 0.0
    return inter_area / union_area

def compute_iou_yolo(box1, box2, image_width, image_height):
    """Compute IoU between two boxes in YOLO format [cls, cx, cy, w, h]."""
    # convert to pixel format
    _, cx1, cy1, w1, h1 = box1
    _, cx2, cy2, w2, h2 = box2

    x1_1 = (cx1 - w1 / 2) * image_width
    y1_1 = (cy1 - h1 / 2) * image_height
    x2_1 = (cx1 + w1 / 2) * image_width
    y2_1 = (cy1 + h1 / 2) * image_height

    x1_2 = (cx2 - w2 / 2) * image_width
    y1_2 = (cy2 - h2 / 2) * image_height
    x2_2 = (cx2 + w2 / 2) * image_width
    y2_2 = (cy2 + h2 / 2) * image_height

    return compute_iou([x1_1, y1_1, x2_1, y2_1], [x1_2, y1_2, x2_2, y2_2])