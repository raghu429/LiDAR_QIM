#!/usr/bin/env python
import sys
import numpy as np
# import pcl
import rospy
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2, PointField
from random import randint
import struct
import ctypes
import ntpath
import std_msgs.msg

from helper_functions import *
from QIM_helper import *
from dither_randomscratchpad import *


# Global Initialization

#set the directory name for the encoded point clouds
pc_dir_path = os.path.join("./QIM_data", "encoded_Dither")
#print('pc dir path', pc_dir_path)
#if that directory doesn't already exists create one
if not os.path.exists(pc_dir_path):
  os.mkdir(pc_dir_path)


#set the directory name for the camera filtered point clouds
clean_pc_dir_path = os.path.join("./QIM_data", "camfiltered")
#print('pc dir path', pc_dir_path)
#if that directory doesn't already exists create one
if not os.path.exists(clean_pc_dir_path):
  os.mkdir(clean_pc_dir_path)

# x = (0,1.4)
# y = (0,1.2)
# z = (0,1.5)

# # resolution_delta = 0.1
# # resolution_halfdelta = resolution_delta/2.0

# # numbits = 3

# groundplane_level = -2.5


#************************************************
#************************************************
# All globals are initialized in QIM_helper.py



if __name__ == '__main__':

  # ROS node initialization
  rospy.init_node('QIM_encode_dither', anonymous = True)

  # Traverse a given directory and work on each file
  # Directory of Kitti binary file

  data_directory = './QIM_data/test_data/'
  #block size in points per frame
  blockSize_list = ['256', '128', '64', '32', '16','8','4', '2']
  # dither range in terms of step/'the number in the list'
  ditherRange_list = ['2','3','4','8']  

  for filename in os.listdir(data_directory):
    
    
    # if filename.endswith(".bin")  and filename.startswith("000071"):
    if filename.endswith(".bin"):
    
      working_file_name = ntpath.basename(os.path.join(data_directory, filename)).split('.')[0]
      print('file currently working on %s'%(working_file_name) )

      qim_encoded_pointcloud = []
      #****************************************************************************************************************************
                                                      # Read & poreprocess point cloud
      #****************************************************************************************************************************

      points_list = load_pc_from_bin(os.path.join(data_directory, filename))
      pc_camera_angle_filtered  = filter_camera_angle(points_list[:,:3])

      pc_groundplane_filtered = filter_groundplane(np.copy(pc_camera_angle_filtered), groundplane_level)

      working_file_name_pcd = working_file_name +'.npy'
      # save corresponding clean point cloud
      np.save(os.path.join(clean_pc_dir_path, working_file_name_pcd), pc_groundplane_filtered)
  
      pc_input = pc_groundplane_filtered
      print('point cloud shape', pc_input.shape)
      
      #****************************************************************************************************************************
                                                  # Encode the point cloud
      #****************************************************************************************************************************
    for j in range(len(blockSize_list)):
        filename_additions_one = []
        block_size = int(blockSize_list[j])
        print('block size', block_size)
        # subdir_path = pc_dir_path + '/' + 'bs_' + blockSize_list[j]
        filename_additions_one = 'bs' + blockSize_list[j]
        for k in range(len(ditherRange_list)):
            filename_additions_two = []
            filename_additions_two = filename_additions_one + '_' + 'dr' + ditherRange_list[k]
            
            # print('output file name and path', op_dir) 
            # Encode the point cloud
            range_factor = int(ditherRange_list[k])
            print('rage factor', range_factor)
            # qim_encode_dither(pc_input, resolution_delta, rate, range_factor)
            qim_encoded_pointcloud = qim_encode_dither(np.copy(pc_input), resolution_delta, block_size, range_factor)    

            print('encoded pc shape', qim_encoded_pointcloud.shape)

            # if not os.path.exists(op_dir):
                # os.makedirs(op_dir)

            filename_additions_two = filename_additions_two + '_' + working_file_name_pcd
            # save encoded clean point cloud
            np.save(os.path.join(pc_dir_path, filename_additions_two), qim_encoded_pointcloud)
          
      #****************************************************************************************************************************
                                                  # Distortion
      #****************************************************************************************************************************

      # distortion = Hausdorff_dist(pc_groundplane_filtered, encoded_pointcloud )
      # print('distortion', distortion)

       

  i = 0

  # Spin while node is not shutdown
  while not rospy.is_shutdown():
  # while rospy.is_shutdown():
    # read pcl and the point cloud
    
    #raw point clouds
       
    # if(len(qim_encoded_pointcloud) !=0):
    #   publish_pc2(qim_encoded_pointcloud, "/encode_qim_encoded_pointcloud")
    # else:
    #   rospy.logerr("%s in encode is empty", "qim_encoded_pointcloud")

    # if(len(pc_camera_angle_filtered) !=0):
    #   publish_pc2(pc_camera_angle_filtered, "/encode_pc_camera_angle_filtered")
    # else:
    #   rospy.logerr("%s in encode is empty", "pc_camera_angle_filtered")

    # if(len(pc_groundplane_filtered) !=0):
    #   publish_pc2(pc_groundplane_filtered, "/encode_pc_groundplane_filtered")
    # else:
    #   rospy.logerr("%s in encode is empty", "pc_groundplane_filtered")
    
    
    print('spin count', i)
    i += 1
  rospy.spin()
