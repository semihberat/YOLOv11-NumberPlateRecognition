# Project Overview: Turkish License Plate Detection and OCR

kaggle: https://www.kaggle.com/code/semihberaterdoan/license-plate-recognition-with-yolov11m

## Introduction
This project is designed to detect Turkish license plates in images and extract their text using a combination of YOLO (You Only Look Once) object detection and Tesseract OCR. The workflow includes dataset preparation, model training, and visualization of results, making it a comprehensive solution for license plate recognition tasks.

## Key Components

### 1. Dataset Preparation
- **Script:** `split_images_and_labels`
- **Purpose:** Splits a dataset of images and labels into training and validation sets.
- **Key Features:**
  - Automatically creates `train` and `val` directories.
  - Ensures proper alignment of images and their corresponding labels.

### 2. YOLO Model Training
- **Script:** YOLO training script in `yolo_training_documentation.md`
- **Purpose:** Trains a YOLO model for license plate detection.
- **Key Features:**
  - Configurable parameters like `epochs`, `batch size`, and `image size`.
  - Supports GPU acceleration for faster training.
  - Includes early stopping and validation.

### 3. License Plate Detection and OCR
- **Script:** `main.py`
- **Purpose:** Detects license plates in images and extracts text using Tesseract OCR.
- **Key Features:**
  - Preprocessing steps like cropping, grayscale conversion, and thresholding.
  - Formats extracted text to match Turkish license plate standards.
  - Visualizes results with bounding boxes and recognized text.

### 4. Metrics Evaluation
- **Notebook:** `mAP_and_IOU_explanation.ipynb`
- **Purpose:** Explains Intersection over Union (IoU) and Mean Average Precision (mAP) metrics.
- **Key Features:**
  - LaTeX-rendered mathematical equations.
  - Visual examples to illustrate concepts.

## Workflow

### Step 1: Dataset Preparation
Use the `split_images_and_labels` function to organize your dataset. Example:
```python
split_images_and_labels(
    val_size=0.2,
    input_dir="/path/to/input",
    output_dir="/path/to/output"
)
```

### Step 2: Train the YOLO Model
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

### Step 3: Detect License Plates and Extract Text
Run the `main.py` script to detect license plates and extract text. Example:
```python
for img_path in random_images:
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = model(img_path, conf=0.4, iou=0.4)
    # Further processing as described in the script
```

### Step 4: Evaluate Metrics
Use the `mAP_and_IOU_explanation.ipynb` notebook to understand IoU and mAP metrics. These metrics are crucial for evaluating the performance of object detection models.

## Key Concepts

### Intersection over Union (IoU)
IoU measures the overlap between predicted and ground truth bounding boxes:
IoU = \frac{Area\ of\ Overlap}{Area\ of\ Union} 

### Mean Average Precision (mAP)
mAP evaluates object detection performance by calculating the average precision for each class and taking the mean:
mAP = \frac{1}{N} \sum_{i=1}^{N} AP_i 

### YOLO
YOLO (You Only Look Once) is a fast and accurate object detection algorithm. It processes the entire image in a single forward pass, predicting bounding boxes and class probabilities simultaneously.

## Example Output
For an image with a detected license plate, the script outputs:
- The processed image with bounding boxes and recognized text.
- The recognized license plate text printed to the console:
  ```
  Plate 1: 34ABC123
  ```

## Notes
- Ensure that the YOLO model weights and Tesseract OCR are correctly installed and configured.
- Adjust parameters like `confidence` and `IoU thresholds` for optimal results.
- The project is optimized for Turkish license plates but can be adapted for other formats with minor modifications.

## Conclusion
This project provides a robust framework for license plate detection and OCR, combining state-of-the-art object detection with effective text recognition techniques. It is suitable for real-world applications like traffic monitoring and automated toll systems.
