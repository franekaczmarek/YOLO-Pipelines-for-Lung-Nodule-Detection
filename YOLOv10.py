import os
from ultralytics import YOLO

class ObjectDetection:
    def __init__(self):
        # Directory where 'dataconfig.yml' is stored
        self.dir = "/home/u154686/Thesis/Scripts/YOLOv10"
        # Path to the YAML configuration file
        self.yaml_path = os.path.join(self.dir, 'dataconfig.yml')

    def train(self, runName):
        # Load YOLOv10 model with custom weights or configuration
        model = YOLO("/home/u154686/Thesis/Scripts/YOLOv10/weights/yolov10l.pt")  # or "yolov10s.yaml" if starting from scratch

        # Start training
        model.train(
            data=self.yaml_path,                    # Path to your dataset config file
            epochs=100,                             # Number of training epochs
            batch=32,                               # Batch size
            imgsz=502,                              # Input image size
            optimizer='SGD',                        # Optimizer
            single_cls=True,                        # Single-class training
            project='train_results',                # Output directory for results
            name=runName,                           # Experiment name
            patience=20,                            # Early stopping patience
            lr0=0.01,                               # Initial learning rate
            lrf=0.01,                               # Final learning rate fraction
            momentum=0.937,                         # Momentum
            weight_decay=0.0005,                    # Weight decay
            warmup_epochs=3.0,                      # Warmup epochs
            warmup_momentum=0.8,                    # Warmup initial momentum
            warmup_bias_lr=0.1,                     # Warmup initial bias lr
            box=0.05,                               # Box loss gain
            cls=0.5,                                # Class loss gain
            iou=0.2,                                # IoU training threshold
            hsv_h=0.015,                            # HSV-Hue augmentation
            hsv_s=0.7,                              # HSV-Saturation augmentation
            hsv_v=0.4,                              # HSV-Value augmentation
            degrees=0.0,                            # Rotation augmentation
            translate=0.1,                          # Translation augmentation
            scale=0.5,                              # Scale augmentation
            shear=0.0,                              # Shear augmentation
            perspective=0.0,                        # Perspective augmentation
            flipud=0.0,                             # Flip up-down augmentation
            fliplr=0.5,                             # Flip left-right augmentation
            mosaic=1.0,                             # Mosaic augmentation
            mixup=0.0,                              # Mixup augmentation
            copy_paste=0.0,                         # Copy-paste augmentation
            workers=8,                              # Number of workers for data loading
            exist_ok=True,                          # Overwrite existing project/name directory
            verbose=True,                           # Verbose output
            resume=False,                           # Resume training from last checkpoint
            multi_scale=False                       # Disable multi-scale training
        )
    @staticmethod
    def train_custom_dataset(runName):
        od = ObjectDetection()
        od.train(runName)

# Example usage:
ObjectDetection.train_custom_dataset('l0')
