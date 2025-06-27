import cv2
import os
import glob
import numpy as np
import matplotlib.pyplot as plt

# Paths
img_dir = "Minor_Project_Dataset.v9i.coco-segmentation/valid/images"
label_dir = "Minor_Project_Dataset.v9i.coco-segmentation/valid/labels"

# Class names (must match your data.yaml exactly)
class_names = [
    "Rice", "Roti", "aloo-bhurji", "daal-pulse", "Besan-Chilla", "Biryani",
    "Bundi", "Channa-Daal", "Chilli-paneer", "Chutney", "Daal-pulse", "Gulab-jamun",
    "Halwa", "Jalebi", "Mushroom", "No-waste", "Palak-paneer", "Paneer-bhurji",
    "Papad", "Raita", "Rajma", "Red--chutney", "Rice-with-pulse", "Salad-Chukundar",
    "Salad-Mix", "Salad-kheera", "Salad-onion", "aloo-shimla", "gajar-aloo",
    "kadi-pakoda", "papad", "pudi", "tea"
]

def draw_segmentation(image, label_path):
    h, w = image.shape[:2]
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            cls = int(parts[0])
            segmentation = list(map(float, parts[1:]))

            points = np.array(segmentation).reshape(-1, 2)
            points *= np.array([w, h])
            points = points.astype(np.int32)

            cv2.polylines(image, [points], isClosed=True, color=(0,255,0), thickness=2)
            cv2.putText(image, class_names[cls], tuple(points[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1)

    return image

# Visualize a few
image_paths = glob.glob(os.path.join(img_dir, "*.jpg"))[:5]
for img_path in image_paths:
    base = os.path.splitext(os.path.basename(img_path))[0]
    label_path = os.path.join(label_dir, base + ".txt")

    if not os.path.exists(label_path):
        continue

    image = cv2.imread(img_path)
    annotated = draw_segmentation(image.copy(), label_path)

    # Show using matplotlib
    plt.figure(figsize=(8,8))
    plt.imshow(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
    plt.title(base)
    plt.axis('off')
    plt.show()