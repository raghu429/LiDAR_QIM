#!/usr/bin/env python
import sys
import numpy as np
import pcl
import rospy
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2, PointField
from random import randint
import struct
import ctypes
import ntpath
import std_msgs.msg
#import settings
from helper_functions import *
from QIM_helper import*
from tamper_pc import *




def pctampering_objectaddition_local(cluster_to_copy,yrange, ymax):
    #get the cluster to be replicated
    displacemnet = yrange*3
   
    if (ymax < 0):
        #modify the cluster
        cluster_to_copy[:,1] = cluster_to_copy[:,1] + displacemnet
    else:
        #modify the cluster
        cluster_to_copy[:,1] = cluster_to_copy[:,1] - displacemnet
        
    return cluster_to_copy



resolution = 0.025

if __name__ == '__main__':
    x = (0,1.4)
    y = (0,1.2)
    z = (0,1.5)

    # ROS node initialization
    rospy.init_node('QIM_encode', anonymous = True)

    data_path = './bbox_test/velodyne/000035.bin'
    calib_path = './bbox_test/calib/000035.txt'
    label_path = './bbox_test/label_2/000035.txt'


    #****************************************************************************************************************************
                # Read point cloud
    #****************************************************************************************************************************

    # read the point cloud from pcd file.. here the os.path.join(directory, filename) would be of type "./QIM_data/test_data/xxxx.bin"
    points_list = load_pc_from_bin(data_path)

    pc_camera_angle_filtered  = filter_camera_angle(points_list[:,:3])
    
    # def read_labels(label_path, label_type, calib_path=None, is_velo_cam=False, proj_velo=None)
    places, rotation, sizes = read_labels(label_path, 'txt', calib_path = None , is_velo_cam=False, proj_velo=None)
    gt_corners = visualize_groundtruth(calib_path, label_path).reshape(-1,3)
    print('corners', gt_corners.shape, gt_corners)

    logical_bound, x_range, y_range, z_range, y_max = get_cluster_logicalbound (np.copy(pc_camera_angle_filtered), gt_corners.reshape(-1,8,3)[1])


    # generate bounding box for the new points based on the points
    generated_boundingbox = get_boundingboxcorners(pctampering_objectaddition_local(np.copy(gt_corners.reshape(-1,8,3)[1]), y_range, y_max))
    
    
    
    bb_a = generated_boundingbox.reshape(-1,3)
    print("bb_a shape", bb_a.shape)
    # generated_boundingbox = np.array([bb_a[2], bb_a[3], bb_a[6], bb_a[7]])
    generated_boundingbox = np.array([bb_a[0], bb_a[7]])
    # (np.copy(pc_camera_angle_filtered[logical_bound]))

    #generate bounding box based on shifting the co-ordinates from original label (this is like ground truth)
    gt_boundingbox = pctampering_objectaddition_local(np.copy(gt_corners.reshape(-1,8,3)[1]), y_range, y_max)

    bb_a = gt_boundingbox.reshape(-1,3)
    # gt_boundingbox = np.array([bb_a[0], bb_a[2], bb_a[3], bb_a[4]])
    # gt_boundingbox = np.array([bb_a[3], bb_a[5]])
    gt_boundingbox = np.array([bb_a[5], bb_a[3]])

    print("gt bounding box", gt_boundingbox.shape, gt_boundingbox)
    print("generated bounding box", generated_boundingbox.shape, generated_boundingbox)

    print("gt bounding box", gt_boundingbox.shape, gt_boundingbox[:,:2])
    print("generated bounding box", generated_boundingbox.shape, generated_boundingbox[:,:2])

    overlap_area = bb_intersection_over_union(generated_boundingbox[:,:2], gt_boundingbox[:,:2])
    print("IOU", overlap_area)
    
    updated_point_cloud, cluster_center = pctampering_objectaddition(np.copy(pc_camera_angle_filtered), np.copy(pc_camera_angle_filtered), logical_bound, x_range, y_range, z_range, y_max)
    
    i = 0
    # Spin while node is not shutdown
    while not rospy.is_shutdown():
        # while rospy.is_shutdown():
        # read pcl and the point cloud

        #raw point clouds

        if(len(pc_camera_angle_filtered) !=0):
            publish_pc2(pc_camera_angle_filtered, "/boundingbox_pointcloud")
        else:
            rospy.logerr("%s in encode is empty", "boundingbox_pointcloud")

        if(len(gt_corners) !=0):
            publish_pc2(gt_corners, "/boundingbox_corners")
        else:
            rospy.logerr("%s in encode is empty", "boundingbox_corners")

        if(len(updated_point_cloud) !=0):
            publish_pc2(updated_point_cloud, "/updated_pointcloud")
        else:
            rospy.logerr("%s in encode is empty", "updated_pointcloud")

        if(len(generated_boundingbox) !=0):
            publish_pc2(generated_boundingbox, "/generated_boundingbox")
        else:
            rospy.logerr("%s in encode is empty", "generated_boundingbox")

        if(len(gt_boundingbox) !=0):
            publish_pc2(gt_boundingbox, "/gt_boundingbox")
        else:
            rospy.logerr("%s in encode is empty", "gt_boundingbox")

        

        print('spin count', i)
        i += 1
    rospy.spin()