# Complex-YOLO-ROS-3D-Object-Detection

## Description: 
The Complex YOLO ROS 3D Object Detection project is an integration of the [Complex YOLOv4](https://github.com/maudzung/Complex-YOLOv4-Pytorch) package into the ROS (Robot Operating System) platform, aimed at enhancing real-time perception capabilities for robotics applications. Using 3D object detection techniques based on Lidar data, the project enables robots and autonomous systems to accurately detect and localize objects in a 3D environment, crucial for safe navigation, obstacle avoidance, and intelligent decision-making.

## Key Features:
1. ROS Integration: Custom ROS nodes for publishing and subscribing to critical data streams.
2. Lidar BEV Images: Lidar Bird's Eye View (BEV) images for a comprehensive 3D representation of the environment.
3. Ground Truth Targets: Accurate ground truth targets for training and evaluation purposes.
4. Complex YOLO Model: Utilization of the "Complex YOLO" architecture, a state-of-the-art 3D object detection model.
5. Real-time Inference: Efficient PyTorch-based model inference to achieve real-time processing.

## Table of Contents
- [Motivation](#motivation)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)


## Motivation
The project's goal is to empower robots and autonomous vehicles with robust and real-time perception capabilities, crucial for real-world applications such as autonomous navigation, object tracking, and dynamic environment interaction. By leveraging advanced 3D object detection algorithms within the ROS ecosystem, the Complex YOLO ROS project opens new horizons for safer and more efficient robotics in dynamic and challenging environments. 

## Installation
Python 3  

ROS noetic

Torch - 2.0.1+cu117

OpenCV

Matplotlib

CV Bridge

## Usage (Check out in [YouTube](https://www.youtube.com/watch?v=roS3FgU9A5E))
1) Clone the Complex YOLO ROS 3D Object Detection repository:
   * `git clone https://github.com/GutlapalliNikhil/Complex-YOLO-ROS-3D-Object-Detection.git`
   * `cd Complex-YOLO-ROS-3D-Object-Detection/src/complex_yolo_ros`
2) Download the 3D KITTI detection dataset from the [KITTI website](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d).
3) Create the folder dataset and create another folder kitti inside that
   * `mkdir folder`
   * `cd folder`
   * `mkdir kitti`
4) Extract the dataset and place the files in the "dataset/kitti" folder with the following structure:
- ImageSets
  - test.txt
  - train.txt
  - val.txt
- testing
  - calib
  - images_2
  - velodyne
- training
  - calib
  - images_2
  - label_2
  - velodyne
5) Create the folder "Checkpoints" inside complex_yolo_ros package and place [this](https://drive.google.com/drive/folders/1RHD9PBvk-9SjbKwoi_Q1kl9-UGFo2Pth?usp=sharing) pretrained weights file inside that.
6) Go back to the workspace directory:
`cd Complex-YOLO-ROS-3D-Object-Detection`
7) Build the ROS packages
`catkin_make`
8) Set up the environment:
`source devel/setup.bash`
9) To publish data related to the velodyne, GT targets, and file names, run:
`rosrun complex_yolo_ros kitti_data_publisher.py`
This will publish the following topics:

* /img_files_name: File names of the images.
* /input_img: BEV images that are input to the neural network.
* /gt_targets: Ground truth labels.

10) Subscribe to the /input_img topic, pass the data to the neural network model, and publish the model outputs on the /predicted_targets topic:
`rosrun complex_yolo_ros kitti_data_subscriber.py`

11) To visualize the outputs and ground truths, run:
`rosrun complex_yolo_ros kitti_data_visualizer.py`

You can see the control on the terminal where kitti_data_publisher.py, where it asks for the command 'n' to run and display next image and 'e' to exit.

This will display the camera view and BEV for both the model's predictions and the ground truth labels.

Now, your Complex YOLO ROS 3D Object Detection project is set up, and you can evaluate the model's performance and visualize the results using ROS.

## Results

FPS obtained on GeForce RTX 3050: 26 fps

#### Individual mAP

* Class 0 (Car): precision = 0.9117, recall = 0.9753, AP = 0.9688, f1: 0.9424
* Class 1 (Ped): precision = 0.6961, recall = 0.9306, AP = 0.7854, f1: 0.7964
* Class 2 (Cyc): precision = 0.8000, recall = 0.9377, AP = 0.9096, f1: 0.8634

#### Overall mAP
mAP: 0.8879181553846359

#### Visualization Demo
![a](https://github.com/GutlapalliNikhil/Complex-YOLO-ROS-3D-Object-Detection/assets/33520288/a0a5abde-604b-4ec2-852b-72756c9a660e)

#### RQT Graph
![b](https://github.com/GutlapalliNikhil/Complex-YOLO-ROS-3D-Object-Detection/assets/33520288/8bc3fab0-9222-4129-91bb-09233eb70a4f)

## Credits
Thanks to the Authors of Complex YOLOv4 for their contribution in the field of 3D Perseption and thanks to the ROS family.

Original Repo: [Complex-YOLOv4-Pytorch](https://github.com/maudzung/Complex-YOLOv4-Pytorch)
