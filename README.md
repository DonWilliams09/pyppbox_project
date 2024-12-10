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
detector: ...
tracker: ...
reider: ...
```

### **Project Structure**

```
├── project/
│   ├── cfg/  # Basic configuration  
│   ├── my_config/ # Project configurations
│   ├── data/ # Store videos there
│   ├── comparebox.py/  
│   ├── toolbox.py/
│ 
└── README.md
```

### **Setup**

1. Clone the repository:
   ```
    git clone https://github.com/DonWilliams09/pyppbox_project.git
    cd project
   ```
2. Place the `.pkl` files provided by the first team in the appropriate REID module directory.
3. Configure paths in the `examples/configs` directory to match your setup.
4. Add the video you want to process in the `data` file

### **Workflow**

The first team focuses on registering new people and training a ReID module. These trained models are saved in `.pkl` files and provided to the tracking team. The tracking team uses these `.pkl` files to perform the tracking and re-identification tasks, integrating them into the pipeline with detectors and trackers.

## **Usage**
1. *Toolbox File*

To run the detection, tracking, and re-identification process, use the following command:
```
python toolbox.py
```
In toolbox.py, you can specify the the configuration used by the detectors and trackers.
```
setConfigDir(config_dir="your_config", load_all=True)
```


2. *Compare Toolbox*

To compare the detection, tracking, and re-identification processes using two different use the following command:
```
python comparebox.py
```
In toolbox.py, you can specify the paths for the configuration directories used by the detectors and trackers.
```
config_dir_one = "/path/to/config_one"
config_dir_two = "/path/to/config_two"
ppb_task(config_dir_one, config_dir_two)
```

3. *Input File*

You can run the toolbox either on a pre-saved video
```
input_video = "path_to_the_video"
cap = cv2.VideoCapture(input_video)
```

or in real-time using a connected camera.
```
cap = cv2.VideoCapture(0)
```
The output will display the processed video stream, where detected and re-identified people will be labeled with their IDs. If the toolbox fails to recognize a person, they will be required to register for inclusion in the ReID module's training. Once completed, the .pkl file will be updated in the toolbox, enabling the person to be identified in future detections.

