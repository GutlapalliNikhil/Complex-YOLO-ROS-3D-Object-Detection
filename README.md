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
- [Contributing](#contributing)
- [License](#license)

## Motivation
The project's goal is to empower robots and autonomous vehicles with robust and real-time perception capabilities, crucial for real-world applications such as autonomous navigation, object tracking, and dynamic environment interaction. By leveraging advanced 3D object detection algorithms within the ROS ecosystem, the Complex YOLO ROS project opens new horizons for safer and more efficient robotics in dynamic and challenging environments.

## Installation
Python 3  

ROS noetic

Torch - 2.0.1+cu117

OpenCV

Matplotlib

CV Bridge

## Usage
1) Create a new workspace directory:

`mkdir {Workspace}`

`cd {Workspace}`
