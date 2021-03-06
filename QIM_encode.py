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
pc_dir_path = os.path.join("./QIM_data", "encoded_QIM")
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
  rospy.init_node('QIM_encode', anonymous = True)

  # Traverse a given directory and work on each file
  # Directory of Kitti binary file

  data_directory = './QIM_data/test_data/'

  for filename in os.listdir(data_directory):
    if filename.endswith(".bin")  and filename.startswith("000071"):
    # if filename.endswith(".bin"):
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
      # c = np.around(((np.copy(pc_input) - np.array([x[0],y[0],z[0]])) / resolution_delta)).astype(np.int32)
      # print('quantized input pc numpy representation', c)

      # quantized_pc  = c*resolution + np.array([x[0],y[0],z[0]])
      # print('quantized input pc values', quantized_pc) 

      # Encode the point cloud
      # #block_size = 8
      # #qim_encoded_pointcloud = qim_encode_dither(np.copy(pc_input), resolution_delta, #block_size)    
      
      if(numbits == 3):
        voxel_delta, voxel_halfdelta = qim_quantize_restricted_threebits_new( np.copy(pc_input))
      elif(numbits == 2):
        voxel_delta, voxel_halfdelta = qim_quantize_restricted_twobits( np.copy(pc_input))
      elif(numbits == 1):
        voxel_delta, voxel_halfdelta = qim_quantize_restricted_onebit( np.copy(pc_input))
      else:
        print('ERROR: INVALID noof bits')

      voxel_halfdelta_npy = np.array([voxel_halfdelta]).reshape(-1,3)
      print('quant encoded shape', voxel_halfdelta_npy.shape)

      qim_encoded_pointcloud = getPointCloud_from_quantizedValues(  np.copy(voxel_halfdelta_npy), resolution_halfdelta, x,y,z)


      print('encoded pc shape', qim_encoded_pointcloud.shape)

      # save encoded clean point cloud
      np.save(os.path.join(pc_dir_path, working_file_name_pcd), qim_encoded_pointcloud)
  
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
       
    if(len(qim_encoded_pointcloud) !=0):
      publish_pc2(qim_encoded_pointcloud, "/encode_qim_encoded_pointcloud")
    else:
      rospy.logerr("%s in encode is empty", "qim_encoded_pointcloud")

    if(len(pc_camera_angle_filtered) !=0):
      publish_pc2(pc_camera_angle_filtered, "/encode_pc_camera_angle_filtered")
    else:
      rospy.logerr("%s in encode is empty", "pc_camera_angle_filtered")

    if(len(pc_groundplane_filtered) !=0):
      publish_pc2(pc_groundplane_filtered, "/encode_pc_groundplane_filtered")
    else:
      rospy.logerr("%s in encode is empty", "pc_groundplane_filtered")
    
    
    print('spin count', i)
    i += 1
  rospy.spin()
