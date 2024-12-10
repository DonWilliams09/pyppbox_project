# GOFFIN Gaspard, LOMBARDO Théo, NZITAT William

<img src="https://raw.githubusercontent.com/rathaROG/screenshot/master/pyppbox/pyppbox_new_wide.png"><br />

# **Person Detection, Tracking, and Re-Identification Toolbox**

## **Overview**

This project is a modular toolbox designed for detecting, tracking, and re-identifying people in various environments. It combines state-of-the-art detection algorithms, tracking methods, and re-identification (REID) modules to enable robust person identification in real-world applications.

## **Features**
- **Detection**: Supports YOLO Classic, YOLO Ultralytics, and Ground Truth-based detection.
- **Tracking**: Implements SORT, DeepSORT, and Centroid-based tracking.
- **Re-Identification**: Integrates Facenet and TorchReid for re-identification using pre-trained `.pkl` files.
- **Modularity**: Designed to allow seamless integration of different detectors, trackers, and REID modules.
- **Multithreading**: Capable of processing multiple streams simultaneously for efficient performance.

---

## **Configurations**

### **Detectors**

| **Parameter**           | **YOLO Classic**                                                                                               | **YOLO Ultralytics**                                                                                     | **GT**                                   |
|--------------------------|---------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------|------------------------------------------|
| `dt_name`               | YOLO_Classic                                                                                                 | YOLO_Ultralytics                                                                                         | GT                                       |
| `nms`                   | 0.45                                                                                                         | -                                                                                                        | -                                        |
| `conf`                  | 0.5                                                                                                          | 0.5                                                                                                      | -                                        |
| `class_file`            | Path to class file (e.g., `coco.names`)                                                                      | -                                                                                                        | -                                        |
| `model_cfg_file`        | Path to YOLO configuration file (e.g., `yolov4.cfg`)                                                         | -                                                                                                        | -                                        |
| `model_weights`         | Path to YOLO weights file (e.g., `yolov4.weights`)                                                           | -                                                                                                        | -                                        |
| `model_image_size`      | 416                                                                                                          | 416                                                                                                      | -                                        |
| `repspoint_calibration` | 0.25                                                                                                         | 0.25                                                                                                     | -                                        |
| `iou`                   | -                                                                                                            | 0.7                                                                                                      | -                                        |
| `imgsz`                 | -                                                                                                            | 416                                                                                                      | -                                        |
| `max_det`               | -                                                                                                            | 100                                                                                                      | -                                        |
| `gt_file`               | -                                                                                                            | -                                                                                                        | Path to ground truth file                |
| `gt_map_file`           | -                                                                                                            | -                                                                                                        | Path to ground truth map file            |

---

### **Trackers**

| **Parameter**           | **Centroid**       | **SORT**                          | **DeepSORT**                       |
|--------------------------|--------------------|------------------------------------|-------------------------------------|
| `tk_name`               | Centroid          | SORT                               | DeepSORT                            |
| `max_spread`            | 64                | -                                  | -                                   |
| `max_age`               | -                 | 1                                  | -                                   |
| `min_hits`              | -                 | 3                                  | -                                   |
| `iou_threshold`         | -                 | 0.3                                | -                                   |
| `nn_budget`             | -                 | -                                  | 100                                 |
| `nms_max_overlap`       | -                 | -                                  | 0.5                                 |
| `max_cosine_distance`   | -                 | -                                  | 0.1                                 |
| `model_file`            | -                 | -                                  | Path to DeepSORT model file         |

---

### **Re-Identification (REID)**

| **Parameter**           | **Facenet**                                                                                                 | **TorchReid**                                                                                                 |
|--------------------------|------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------|
| `ri_name`               | Facenet                                                                                                    | Torchreid                                                                                                     |
| `gpu_mem`               | 0.585                                                                                                      | -                                                                                                              |
| `model_det`             | Path to Facenet detection model                                                                            | -                                                                                                              |
| `model_file`            | Path to Facenet model file                                                                                 | Path to TorchReid model file                                                                                   |
| `classifier_pkl`        | Path to classifier `.pkl` file                                                                             | Path to TorchReid classifier `.pkl` file                                                                       |
| `train_data`            | Path to training dataset                                                                                   | Path to training dataset                                                                                       |
| `batch_size`            | 1000                                                                                                       | -                                                                                                              |
| `min_confidence`        | 0.75                                                                                                       | 0.35                                                                                                           |
| `yl_h_calibration`      | [-125, 75]                                                                                                 | -                                                                                                              |
| `yl_w_calibration`      | [-55, 55]                                                                                                  | -                                                                                                              |
| `device`                | -                                                                                                          | `cpu` or `gpu`                                                                                                |

---

### **Main Configuration**

Set the main modules in the `main_config.yaml` file:

```yaml
# Main config:
###########################################################
# detector: None | YOLO_Classic | YOLO_Ultralytics
# tracker: None | Centroid | SORT | DeepSORT
# reider: None | Facenet | Torchreid
###########################################################
detector: YOLO_Ultralytics
tracker: DeepSORT
reider: Facenet

