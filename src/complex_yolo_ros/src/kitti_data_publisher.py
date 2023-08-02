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
import keyboard

import matplotlib.pyplot as plt
from std_msgs.msg import String, Float32MultiArray
from sensor_msgs.msg import Image
import torch
from torch.utils.data import DataLoader
#from complex_yolo_src.msg import TargetArray
sys.path.append('/media/nikhil/Ubuntu_11/pointpillars/ComplexYoloROS/src/complex_yolo_ros/src')

from data_process.kitti_dataset import KittiDataset
from data_process.transformation import Compose, OneOf, Random_Rotation, Random_Scaling, Horizontal_Flip, Cutout

from utils.evaluation_utils import post_processing_v2

import rospy
from cv_bridge import CvBridge
import cv2


def convert_tensor_to_cv2(img_tensor):
    # Convert the PyTorch tensor to a NumPy array and then to a CV2 image
    img_np = img_tensor.cpu().numpy()
    img_np = (img_np * 255).astype('uint8')  # Convert to uint8 format
    img_cv2 = cv2.cvtColor(img_np.transpose(1, 2, 0), cv2.COLOR_RGB2BGR)
    return img_cv2

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


    if configs.show_train_data:
        dataloader, _ = create_train_dataloader(configs)
        print('len train dataloader: {}'.format(len(dataloader)))
    else:
        dataloader = create_val_dataloader(configs)
        print('len val dataloader: {}'.format(len(dataloader)))

    print('\n\nPress n to see the next sample >>> Press Esc to quit...')
    rospy.init_node('kitti_data_publisher', anonymous=True)
    
    img_files_pub = rospy.Publisher('img_files_name', String, queue_size=10)
    imgs_pub = rospy.Publisher('input_img', Image, queue_size=10)
    targets_pub = rospy.Publisher("gt_targets", String, queue_size=10)

    rate = rospy.Rate(10)
   
    print("System Check Done")
    for batch_i, (img_files, imgs, targets) in enumerate(dataloader):
        
        print("###################################################")

        img_files_msg = String(data=img_files[0])
        img_files_pub.publish(img_files_msg)
        
        imgs_cv2 = convert_tensor_to_cv2(imgs[0])

        bridge = CvBridge()
        # Convert the NumPy array to an Image message
        imgs_msg = bridge.cv2_to_imgmsg(imgs_cv2, encoding="bgr8") # Convert to BGR format

        # Set the timestamp (optional)
        imgs_msg.header.stamp = rospy.Time.now()

        # Publish the image
        imgs_pub.publish(imgs_msg)
        
        targets_list = targets.cpu().detach().numpy().tolist()
        targets_msg = String(data=str(targets_list))
        targets_pub.publish(targets_msg)

        rate.sleep()
        
        while True:
            user_input = input("Press 'n' to continue to the next iteration or 'e' to exit: ")
            if user_input.lower() == 'n':
                break
            if user_input.lower() == 'e':
                exit()

