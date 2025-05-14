# **YOLOv11-NumberPlateRecognition**

kaggle: https://www.kaggle.com/code/semihberaterdoan/license-plate-recognition-with-yolov11m

## YOLO Training

## Overview of YOLO
YOLO (You Only Look Once) is a state-of-the-art object detection algorithm that performs detection in a single pass through the network. Unlike traditional methods that use region proposals and multiple stages, YOLO treats object detection as a regression problem, predicting bounding boxes and class probabilities directly from the input image.

### Key Features of YOLO
1. **Speed:** YOLO is extremely fast because it processes the entire image in a single forward pass.
2. **Accuracy:** It achieves high accuracy by learning global image context and spatial relationships.
3. **Unified Architecture:** YOLO uses a single convolutional neural network (CNN) to predict bounding boxes and class probabilities simultaneously.

## Explanation of IoU and mAP Metrics

### Intersection over Union (IoU)
IoU measures the overlap between two bounding boxes, typically the predicted and ground truth boxes. It is defined as:
$$ IoU = \frac{Area\ of\ Overlap}{Area\ of\ Union} $$

- **Area of Overlap:** The area where the predicted and ground truth boxes intersect.
- **Area of Union:** The total area covered by both boxes.

IoU ranges from 0 to 1, where 1 indicates perfect overlap. It is used to determine whether a predicted bounding box is a true positive or a false positive.

### Mean Average Precision (mAP)
mAP evaluates the performance of object detection models by calculating the average precision (AP) for each class and then taking the mean.

#### Average Precision (AP)
AP is the area under the Precision-Recall curve for a specific class. It is calculated as:
$$ AP = \int_0^1 P(R) dR $$

- **P(R):** Precision as a function of Recall.

#### Mean Average Precision
The mAP is then calculated as:
$$ mAP = \frac{1}{N} \sum_{i=1}^{N} AP_i $$

- **N:** Total number of classes.
- **AP_i:** Average Precision for class $i$.

## Explanation of the Training Code

### GPU Configuration
```python
gpu_count = torch.cuda.device_count()
device = list(range(gpu_count)) if gpu_count > 1 else 0
```
- **`torch.cuda.device_count()`:** Checks the number of available GPUs.
- **`device`:** Sets the device to use multiple GPUs if available, otherwise defaults to a single GPU.

### Model Initialization
```python
model = YOLO("yolo11n.pt")
```
- **`YOLO`:** Initializes the YOLO model with the specified weights file (`yolo11n.pt`).

### Training Configuration
```python
results = model.train(
    data="/kaggle/working/data.yaml",   # Dataset configuration
    epochs=200,                         # 200 epochs
    imgsz=640,                          # Suitable for smaller images
    batch=32,                           # Adjustable based on GPU RAM
    workers=2,                          # Ideal starting value for Tesla T4
    device=device,                      # GPU setting
    augment=True,                       # Default YOLO augmentations are automatically enabled
    patience=20,                        # Stops early if no improvement for 20 epochs
    val=True,                           # Validation is performed at the end of each epoch
)
```
#### Key Parameters
- **`data`:** Path to the dataset configuration file (`data.yaml`).
- **`epochs`:** Number of training epochs (200 in this case).
- **`imgsz`:** Image size for training (640x640 pixels).
- **`batch`:** Batch size, adjustable based on GPU memory.
- **`workers`:** Number of data loader workers (2 is ideal for Tesla T4 GPUs).
- **`device`:** Specifies the GPU(s) to use.
- **`augment`:** Enables default YOLO augmentations for data augmentation.
- **`patience`:** Early stopping if no improvement for 20 epochs.
- **`val`:** Enables validation at the end of each epoch.

## Notes
- Ensure that the dataset is correctly formatted and the `data.yaml` file is properly configured.
- Adjust the `batch` size and `workers` based on the available GPU resources.
- Use a pre-trained weights file (e.g., `yolo11n.pt`) to speed up training and improve accuracy.

## Output
The training process outputs:
1. **Model Weights:** Saved at regular intervals and at the end of training.
2. **Metrics:** Training and validation loss, mAP, and other performance metrics.
3. **Logs:** Detailed logs for each epoch, including loss and mAP values.

This documentation provides a comprehensive explanation of the YOLO training process, including the metrics and code used. Let me know if you need further clarifications or additional details!
