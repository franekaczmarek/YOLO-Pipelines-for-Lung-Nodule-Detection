# YOLO-Pipelines-for-Lung-Nodule-Detection
# YOLO Pipelines for Lung Nodule Detection

## Overview

This project aims to develop a robust pipeline for detecting lung nodules in computed tomography (CT) scans using advanced YOLO models. The goal is to improve early detection of lung cancer by leveraging state-of-the-art object detection techniques, specifically comparing YOLOv5 and YOLOv10 architectures. The pipeline integrates preprocessing, segmentation, and augmentation strategies to optimize model performance for clinical applications.

## Key Features

### Lung Segmentation
- Utilizes the R231 model for accurate segmentation of lung regions in CT scans.
- Ensures that analysis focuses on relevant areas, reducing noise and improving detection accuracy.

### Preprocessing Pipeline
- **Windowing**: Adjusts Hounsfield Unit (HU) values to enhance contrast in lung tissues, with optimal parameters (-600 HU center, 1500 HU width).
- **Normalization**: Converts CT images to an 8-bit grayscale format for consistency.
- **Bounding Box Adjustments**: Expands bounding boxes by 30% and adds 5-pixel padding to provide contextual information around nodules.

### Data Augmentation
- Applies rotation, zooming, and cropping techniques to balance class distribution and enrich training data.
- Ensures robust model generalization by creating diverse examples.

### YOLO Model Implementation
- Trains YOLOv5 and YOLOv10 models using a custom dataset derived from the LIDC-IDRI database.
- Configures hyperparameters such as batch size (32), image size (502 pixels), and 100 epochs.
- Compares performance metrics like mAP@0.5, precision, recall, and inference time.

### Error Analysis
- Identifies false positives (misclassified structures) and false negatives (missed nodules), particularly small nodules or those near lung boundaries.
- Highlights challenges in peripheral slices of CT scans.

### Data Splitting
- Implements patient-wise splitting into training (70%), validation (20%), and testing (10%) sets to avoid data leakage and ensure unbiased evaluation.

## Results

| Model       | mAP@0.5 | Precision | Recall | Inference Time |
|-------------|----------|-----------|--------|----------------|
| YOLOv5-l    | 0.54     | 0.62      | 0.53   | 8.6 ms         |
| YOLOv10-l   | **0.61** | **0.66**  | **0.54** | **6.8 ms**     |

- YOLOv10-l outperformed YOLOv5-l in terms of mAP@0.5, precision, recall, and inference time.
- Optimized preprocessing techniques significantly improved detection accuracy.
- Error analysis revealed challenges with small nodules and those near lung boundaries, emphasizing areas for improvement.

## Potential Applications

### Clinical Diagnostics
- Enhances Computer-Aided Diagnosis (CAD) systems for early detection of lung cancer.
- Reduces radiologists' workload by automating nodule detection.

### Epidemiological Studies
- Facilitates large-scale analysis of lung health across populations using high-throughput detection pipelines.

### Educational Tools
- Assists radiologists in understanding AI predictions and improving diagnostic confidence.

## Challenges and Recommendations

1. **Detection Limitations**:
   - Small nodules and nodules near lung boundaries are difficult to detect.
   - Consider higher-resolution input images or specialized loss functions.

2. **Dataset Diversity**:
   - Validate on external datasets like LUNA16 or NLST for better generalization.

3. **3D Context**:
   - Explore volumetric analysis across CT slices for richer spatial context.

4. **Benchmarking**:
   - Compare against other frameworks like Faster R-CNN or Swin Transformers.

## Conclusion

This GitHub project demonstrates the transformative potential of YOLO-based pipelines in medical imaging, particularly for lung nodule detection. By combining advanced preprocessing techniques, robust model architectures, and systematic evaluation strategies, the project contributes to improving CAD systems' accuracy and reliability in clinical settings.

For more details, refer to the code implementation provided in this repository.
