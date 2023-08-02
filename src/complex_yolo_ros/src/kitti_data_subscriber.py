"""
# -*- coding: utf-8 -*-
-----------------------------------------------------------------------------------
# Author: Nguyen Mau Dung
# DoC: 2020.07.05
# email: nguyenmaudung93.kstn@gmail.com
-----------------------------------------------------------------------------------
# Description: This script for creating the dataloader for training/validation/test phase
"""
print("Please Wait untill System Check")
print("System Check in Progress")
import sys
import rospy
from std_msgs.msg import Int32, Float32MultiArray, String
from sensor_msgs.msg import Image
import torch
from torch.utils.data import DataLoader
from cv_bridge import CvBridge

sys.path.append('/media/nikhil/Ubuntu_11/pointpillars/ComplexYoloROS/src/complex_yolo_ros/src')

from data_process.kitti_dataset import KittiDataset
from data_process.transformation import Compose, OneOf, Random_Rotation, Random_Scaling, Horizontal_Flip, Cutout

from models.model_utils import create_model

from utils.evaluation_utils import post_processing_v2
import matplotlib.pyplot as plt


# Global variables to store received data
img_files_msg = None
imgs_msg = None
targets_msg = None

def imgs_callback(data):
    global imgs_msg
    imgs_msg = data

    
def convert_cv2_to_tensor(img_cv2):
    # Convert the CV2 image to a NumPy array and then to a PyTorch tensor
    img_np = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
    img_np = img_np.transpose(2, 0, 1)
    img_tensor = torch.tensor(img_np, dtype=torch.float32) / 255.0  # Convert to float32 and normalize to [0, 1]
    return img_tensor
    
def create_train_dataloader(configs):
    """Create dataloader for training"""

    train_lidar_transforms = OneOf([
        Random_Rotation(limit_angle=20., p=1.0),
        Random_Scaling(scaling_range=(0.95, 1.05), p=1.0)
    ], p=0.66)

    train_aug_transforms = Compose([
        Horizontal_Flip(p=configs.hflip_prob),
        Cutout(n_holes=configs.cutout_nholes, ratio=configs.cutout_ratio, fill_value=configs.cutout_fill_value,
               p=configs.cutout_prob)
    ], p=1.)

    train_dataset = KittiDataset(configs.dataset_dir, mode='train', lidar_transforms=train_lidar_transforms,
                                 aug_transforms=train_aug_transforms, multiscale=configs.multiscale_training,
                                 num_samples=configs.num_samples, mosaic=configs.mosaic,
                                 random_padding=configs.random_padding)
    train_sampler = None
    if configs.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, batch_size=configs.batch_size, shuffle=(train_sampler is None),
                                  pin_memory=configs.pin_memory, num_workers=configs.num_workers, sampler=train_sampler,
                                  collate_fn=train_dataset.collate_fn)

    return train_dataloader, train_sampler


def create_val_dataloader(configs):
    """Create dataloader for validation"""
    val_sampler = None
    val_dataset = KittiDataset(configs.dataset_dir, mode='val', lidar_transforms=None, aug_transforms=None,
                               multiscale=False, num_samples=configs.num_samples, mosaic=False, random_padding=False)
    if configs.distributed:
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False)
    val_dataloader = DataLoader(val_dataset, batch_size=configs.batch_size, shuffle=False,
                                pin_memory=configs.pin_memory, num_workers=configs.num_workers, sampler=val_sampler,
                                collate_fn=val_dataset.collate_fn)

    return val_dataloader


def create_test_dataloader(configs):
    """Create dataloader for testing phase"""

    test_dataset = KittiDataset(configs.dataset_dir, mode='test', lidar_transforms=None, aug_transforms=None,
                                multiscale=False, num_samples=configs.num_samples, mosaic=False, random_padding=False)
    test_sampler = None
    if configs.distributed:
        test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, batch_size=configs.batch_size, shuffle=False,
                                 pin_memory=configs.pin_memory, num_workers=configs.num_workers, sampler=test_sampler)

    return test_dataloader


if __name__ == '__main__':
    import argparse
    import os

    import cv2
    import numpy as np
    from easydict import EasyDict as edict

    import data_process.kitti_bev_utils as bev_utils
    from data_process import kitti_data_utils
    from utils.visualization_utils import show_image_with_boxes, merge_rgb_to_bev, invert_target
    import config.kitti_config as cnf

    parser = argparse.ArgumentParser(description='Complexer YOLO Implementation')

    parser.add_argument('--img_size', type=int, default=608,
                        help='the size of input image')
    parser.add_argument('--hflip_prob', type=float, default=0.,
                        help='The probability of horizontal flip')
    parser.add_argument('--cutout_prob', type=float, default=0.,
                        help='The probability of cutout augmentation')
    parser.add_argument('--cutout_nholes', type=int, default=1,
                        help='The number of cutout area')
    parser.add_argument('--cutout_ratio', type=float, default=0.3,
                        help='The max ratio of the cutout area')
    parser.add_argument('--cutout_fill_value', type=float, default=0.,
                        help='The fill value in the cut out area, default 0. (black)')
    parser.add_argument('--multiscale_training', action='store_true',
                        help='If true, use scaling data for training')

    parser.add_argument('--mosaic', action='store_true',
                        help='If true, compose training samples as mosaics')
    parser.add_argument('--random-padding', action='store_true',
                        help='If true, random padding if using mosaic augmentation')
    parser.add_argument('--show-train-data', action='store_true',
                        help='If true, random padding if using mosaic augmentation')
    parser.add_argument('--output-width', type=int, default=608,
                        help='the width of showing output, the height maybe vary')
    parser.add_argument('--save_img', action='store_true',
                        help='If true, save the images')


    parser.add_argument('--classnames-infor-path', type=str, default='/media/nikhil/Ubuntu_11/pointpillars/ComplexYoloROS/src/complex_yolo_ros/dataset/kitti/classes_names.txt',
                        metavar='PATH', help='The class names of objects in the task')
    parser.add_argument('-a', '--arch', type=str, default='darknet', metavar='ARCH',
                        help='The name of the model architecture')
    parser.add_argument('--cfgfile', type=str, default='/media/nikhil/Ubuntu_11/pointpillars/ComplexYoloROS/src/complex_yolo_ros/src/config/cfg/complex_yolov4.cfg', metavar='PATH',
                        help='The path for cfgfile (only for darknet)')
    parser.add_argument('--pretrained_path', type=str, default="/media/nikhil/Ubuntu_11/pointpillars/ComplexYoloROS/src/complex_yolo_ros/checkpoints/complex_yolov4_mse_loss.pth", metavar='PATH',
                        help='the path of the pretrained checkpoint')
    parser.add_argument('--use_giou_loss', action='store_true',
                        help='If true, use GIoU loss during training. If false, use MSE loss for training')

    parser.add_argument('--no_cuda', action='store_true',
                        help='If true, cuda is not used.')
    parser.add_argument('--gpu_idx', default=0, type=int,
                        help='GPU index to use.')

    parser.add_argument('--num_samples', type=int, default=None,
                        help='Take a subset of the dataset to run and debug')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of threads for loading data')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='mini-batch size (default: 4)')

    parser.add_argument('--conf-thresh', type=float, default=0.5,
                        help='for evaluation - the threshold for class conf')
    parser.add_argument('--nms-thresh', type=float, default=0.5,
                        help='for evaluation - the threshold for nms')
    parser.add_argument('--iou-thresh', type=float, default=0.5,
                        help='for evaluation - the threshold for IoU')
    configs = edict(vars(parser.parse_args()))
    configs.distributed = False  # For testing
    configs.pin_memory = False
    configs.dataset_dir = os.path.join('/media/nikhil/Ubuntu_11/pointpillars/ComplexYoloROS/src/complex_yolo_ros', 'dataset', 'kitti')
    
    model = create_model(configs)
    
    model.load_state_dict(torch.load(configs.pretrained_path, map_location='cuda:0'))

    configs.device = torch.device('cpu' if configs.no_cuda else 'cuda:{}'.format(configs.gpu_idx))
    model = model.to(device=configs.device)

    model.eval()
    


    rospy.init_node('kitti_data_subscriber', anonymous=True)
    
    rospy.Subscriber('input_img', Image, imgs_callback)

    targets_pub_1 = rospy.Publisher("predicted_targets", String, queue_size=10)
    
    rate = rospy.Rate(10)

    print("System Check Done")
    while not rospy.is_shutdown():
      
      if imgs_msg is not None:
      
        print("#######################################")

        
        bridge = CvBridge()
        
        imgs = bridge.imgmsg_to_cv2(imgs_msg, desired_encoding="bgr8")
        
        #cv2.imshow("Received Image", imgs)
        #cv2.waitKey(1)  # Required for OpenCV to update the image display

        # Assuming imgs_cv2 is the OpenCV image
        imgs = convert_cv2_to_tensor(imgs)
        
        imgs = torch.unsqueeze(imgs, 0)
        
        imgs_1 = imgs.to(configs.device, non_blocking=True)
        
        outputs = model(imgs_1)
        outputs = post_processing_v2(outputs, conf_thresh=configs.conf_thresh, nms_thresh=configs.nms_thresh)
        outputs = outputs[0]
        
        
        targets_list_1 = outputs.cpu().detach().numpy().tolist()
        targets_msg_1 = String(data=str(targets_list_1))
        targets_pub_1.publish(targets_msg_1)
        
        print("Model Outputs Published")

        imgs_msg = None 
        targets_msg = None    
        

