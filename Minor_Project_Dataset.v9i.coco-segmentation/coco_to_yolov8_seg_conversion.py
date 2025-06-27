import os
import json
from tqdm import tqdm

def convert_coco_to_yolov8_seg(coco_json_path, image_dir, label_out_dir):
    os.makedirs(label_out_dir, exist_ok=True)

    with open(coco_json_path, 'r') as f:
        coco = json.load(f)

    images = {img['id']: img for img in coco['images']}
    annotations = coco['annotations']

    image_to_annotations = {}
    for ann in annotations:
        img_id = ann['image_id']
        if img_id not in image_to_annotations:
            image_to_annotations[img_id] = []
        image_to_annotations[img_id].append(ann)

    for img_id, anns in tqdm(image_to_annotations.items(), desc=f"Processing {os.path.basename(image_dir)}"):
        img_info = images[img_id]
        file_name = os.path.splitext(img_info['file_name'])[0]
        width = img_info['width']
        height = img_info['height']

        label_path = os.path.join(label_out_dir, f"{file_name}.txt")
        with open(label_path, "w") as f:
            for ann in anns:
                cat_id = ann["category_id"]
                segmentation = ann.get("segmentation", [])

                if not segmentation or not isinstance(segmentation[0], list):
                    continue  # Skip invalid or RLE annotations

                points = segmentation[0]
                if len(points) < 6:
                    continue  # YOLOv8 requires at least 3 points (x, y)

                # Normalize coordinates
                norm_points = []
                for i in range(0, len(points), 2):
                    x = points[i] / width
                    y = points[i+1] / height
                    norm_points.extend([x, y])

                # Format: class_id x1 y1 x2 y2 ...
                f.write(f"{cat_id} " + " ".join([f"{pt:.6f}" for pt in norm_points]) + "\n")


# === Modify these paths ===
train_dir = r"C:\Users\abhin\OneDrive\Documents\GitHub\Albumentationsx\Minor_Project_Dataset.v9i.coco-segmentation\train"
valid_dir = r"C:\Users\abhin\OneDrive\Documents\GitHub\Albumentationsx\Minor_Project_Dataset.v9i.coco-segmentation\valid"

convert_coco_to_yolov8_seg(
    coco_json_path=os.path.join(train_dir, "_annotations.coco.json"),
    image_dir=os.path.join(train_dir, "images"),
    label_out_dir=os.path.join(train_dir, "labels")
)

convert_coco_to_yolov8_seg(
    coco_json_path=os.path.join(valid_dir, "_annotations.coco.json"),
    image_dir=os.path.join(valid_dir, "images"),
    label_out_dir=os.path.join(valid_dir, "labels")
)

print("\n✅ Conversion Complete! You can now train with YOLOv8.")