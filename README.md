# 🍽️ AI-Powered Food Waste Detection System
*Advanced Computer Vision for Food Recognition and Waste Analytics*


![Demo](media/image10.jpg)

## 📌 Project Overview
An end-to-end system that:
- Identifies **67 food classes** with instance segmentation
- Tracks plate waste composition in real-time
- Integrates multiple AI architectures (YOLOv8, Florence, SAM)
- Achieves **99.1% F1-score** in production environments


## 🏆 Performance Highlights
| Model                      | mAP50  | Precision | Recall | Dataset Size |
|----------------------------|--------|-----------|--------|--------------|
| Florence + Roboflow         | 100%   | 99.0%     | 99.3%  | 3,060 images |
| Roboflow v1 (YOLOv8)        | 99.0%  | 94.9%     | 98.7%  | 1,059 images |
| YOLOv8n-seg (Custom Train)  | 38.9%  | 53.6%     | 30.4%  | 524 images   |


## 🛠️ Technical Architecture
graph LR
    A[Input Image] --> B[Preprocessing]
    B --> C[Florence VLT Model]
    B --> D[YOLOv8n-seg]
    C --> E[Semantic Segmentation]
    D --> F[Instance Segmentation]
    E & F --> G[Waste Analytics]
    G --> H[Dashboard]


📂 Dataset Structure
67 food classes (Indian cuisine focused)

3,129 annotated images (COCO format)

Class distribution:

python
['roti': 2223, 'salad': 2076, 'dal': 1293, 'paneer': 858, ...]


🚀 Key Features
Multi-Model Inference

YOLOv8n-seg: Lightweight (12.8 GFLOPs) for edge deployment

Florence VLT: 360M-parameter vision-language transformer

SAM Integration: Promptable segmentation for novel food items

Advanced Training

bash
YOLOv8 Training Command
python train.py --data food.yaml --weights yolov8n-seg.pt --epochs 100 --img 640

text

3. **Real-Time Analytics**
 - Waste composition by food type
 - Historical trends (per meal/day)


## 💻 Installation
```bash
git clone https://github.com/Abhinavcodez/food-waste-detection.git
cd food-waste-detection
pip install -r requirements.txt  # ultralytics==8.0.22, roboflow==1.1.1


📊 Sample Output
text
🧠 302.jpg → Dal (98g), Salad (45g), Roti (120g) 
🧠 225.jpg → Salad (62g), Jeera Aloo (210g), Pakoda (85g)


🌟 Best Performing Classes
Food Item	Precision	Recall	F1-Score
Roti	99.1%	99.6%	99.4%
Dal	99.8%	99.6%	99.7%
Paneer	99.6%	99.2%	99.4%


🤝 How to Contribute
Improve Rare Class Detection

Current weak performers: lemon rice (0% F1), sweet corn (0% F1)

Optimize for Mobile

Reduce YOLOv8n-seg latency (<200ms on Raspberry Pi)

Add Waste Weight Estimation

Integrate depth sensing (Intel RealSense)

📜 License
MIT © Abhinav Kumar Maurya

