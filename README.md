# Turkish License Plate Detection and OCR

## Overview
This repository contains scripts and documentation for detecting Turkish license plates and performing Optical Character Recognition (OCR) using YOLO and Tesseract OCR. The project includes tools for dataset preparation, model training, and visualization of results.

## Features
- **YOLO Object Detection:** Detects license plates in images.
- **Tesseract OCR:** Extracts text from detected license plates.
- **Dataset Splitting:** Splits images and labels into training and validation sets.
- **Visualization:** Displays images with bounding boxes and recognized text.
- **Metrics:** Explains IoU and mAP for evaluating object detection models.

## Getting Started

### Prerequisites
- Python 3.8+
- Required libraries: `cv2`, `pytesseract`, `matplotlib`, `ultralytics`, `glob`
- Tesseract OCR installed and configured.

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/turkish-license-plate-detection.git
   cd turkish-license-plate-detection
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### 1. Dataset Preparation
Use the `split_images_and_labels` function to organize your dataset. Example:
```python
split_images_and_labels(
    val_size=0.2,
    input_dir="/path/to/input",
    output_dir="/path/to/output"
)
```

### 2. Train the YOLO Model
Train the YOLO model using the provided script. Example:
```python
results = model.train(
    data="/kaggle/working/data.yaml",
    epochs=200,
    imgsz=640,
    batch=32,
    workers=2,
    device=device,
    augment=True,
    patience=20,
    val=True,
)
```

### 3. License Plate Detection and OCR
Run the `main.py` script to detect license plates and extract text. Example:
```python
for img_path in random_images:
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = model(img_path, conf=0.4, iou=0.4)
    # Further processing as described in the script
```

### 4. Visualization
Visualize images with bounding boxes and recognized text using Matplotlib. Example:
```python
plt.figure(figsize=(8, 6))
plt.imshow(img_rgb)
plt.axis("off")
plt.show()
```

### 5. Metrics Evaluation
Use the `mAP_and_IOU_explanation.ipynb` notebook to understand IoU and mAP metrics. These metrics are crucial for evaluating the performance of object detection models.

## Key Concepts

### Intersection over Union (IoU)
IoU measures the overlap between predicted and ground truth bounding boxes:
\[
IoU = \frac{\text{Area of Overlap}}{\text{Area of Union}}
\]

### Mean Average Precision (mAP)
mAP evaluates object detection performance by calculating the average precision for each class and taking the mean:
\[
mAP = \frac{1}{N} \sum_{i=1}^{N} AP_i
\]

### YOLO
YOLO (You Only Look Once) is a fast and accurate object detection algorithm. It processes the entire image in a single forward pass, predicting bounding boxes and class probabilities simultaneously.

## Example Output
For an image with a detected license plate, the script outputs:
- The processed image with bounding boxes and recognized text.
- The recognized license plate text printed to the console:
  ```
  Plate 1: 34ABC123
  ```

## Repository Structure
- `main.py`: Main script for license plate detection and OCR.
- `convert_xml2yolo_documentation.md`: Documentation for converting XML annotations to YOLO format.
- `split_images_and_labels_documentation.md`: Documentation for splitting datasets into training and validation sets.
- `yolo_training_documentation.md`: Detailed explanation of YOLO training parameters and process.
- `license_plate_detection_documentation.md`: Documentation for the license plate detection and OCR workflow.
- `mAP_and_IOU_explanation.ipynb`: Notebook explaining IoU and mAP metrics with LaTeX-rendered equations.

## Contributing
Contributions are welcome! Please fork the repository and submit a pull request.

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.
