#!/usr/bin/env python
import sys
import numpy as np
import tensorflow as tf
from helper_functions import *
import glob
import os
import pcl


if __name__ == '__main__':

    #initialize node
    rospy.init_node("lidardata_publisher")
    
    
    #read the data file
    dataformat  =  "bin";
    data_path = "./data/002937.bin"

    if dataformat == "bin":
        pc = load_pc_from_bin(data_path)
    elif dataformat == "pcd":
        pc = load_pc_from_pcd(data_path)

    #display data file at various stages
    #1. see raw point cloud
    #publish_pc2_gt(pc, "/pointcloud_raw")
    
    
    #2. visualize voxel
    resolution = 0.2
    voxel_shape=(800, 800, 40)
    x=(0, 80)
    y=(-40, 40)
    z=(-2.5, 1.5)
    
    #pc_logicalbounds = apply_logicalbounds_topc(pc, x=x, y=y, z=z)
    logical_bounds = get_pc_logicaldivision(pc, x=x, y=y, z=z)
    pc_logicalbounds = apply_logicalbounds_topc(pc, logical_bounds)
    pc_quantized = quantize_pc(pc_logicalbounds, resolution=resolution, x=x, y=y, z=z, quant='low')
    voxel =  draw_to_voxel(pc_quantized, resolution=resolution, x=x, y=y, z=z)

    print('voxel shape before any modificatins', voxel.shape)
    print('voxel before modification first 25 elements', voxel[:25,:])
    #publish_pc2_gt(voxel, "/pointcloud_voxelized")

    #3 print('entering camera angle')
    pc_camfiltered = filter_camera_angle_groundplane(pc[:,:3])
    #publish_pc2_gt(pc_camfiltered, "/pointcloud_camerafiltered")

    
    # # Downsample the cloud as high resolution which comes with a computation cost
    # pcl_data = pcl.PointCloud_PointXYZRGB
    # pcl_data_list  = []

    # print('pc logical bounds length', pc_logicalbounds.shape[0])
    # for data in range(0, pc_logicalbounds.shape[0]):
    #     pcl_data_list.append([pc_logicalbounds[data,0], pc_logicalbounds[data,1], pc_logicalbounds[data,2]])

    # print('pcl logical bounds shape', pc_logicalbounds.shape)
    # pcl_data.from_list(pcl_data_list)
    # voxelgrid_cloud = do_voxel_grid_filter(point_cloud = pcl_data, LEAF_SIZE = 0.1)
    
    # print('after logical anding pc shape', size(voxelgrid_cloud))
    # print('after logical anding first 25 elements', voxelgrid_cloud[:25,:])
    
    count = 0
    while not rospy.is_shutdown():
        
        print count
        publish_pc2(pc, "/pointcloud_raw")
        publish_pc2(pc_logicalbounds, "/pointcloud_logicalbounds")
        publish_pc2(pc_camfiltered, "/pointcloud_camerafiltered")

        count += 1

    rospy.spin()