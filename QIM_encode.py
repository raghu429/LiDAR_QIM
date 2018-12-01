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
from kitti_clustering import *
from tamper_pc import *
from QIM_helper import *


# Global Initialization

#set the directory name for the encoded point clouds
pc_dir_path = os.path.join("./data", "encoded")
#print('pc dir path', pc_dir_path)
#if that directory doesn't already exists create one
if not os.path.exists(pc_dir_path):
  os.mkdir(pc_dir_path)


#set the directory name for the camera filtered point clouds
clean_pc_dir_path = os.path.join("./data", "camfiltered")
#print('pc dir path', pc_dir_path)
#if that directory doesn't already exists create one
if not os.path.exists(clean_pc_dir_path):
  os.mkdir(clean_pc_dir_path)


#set the directory name for meta data
meta_dir_path = os.path.join("./data", "metadata")
#print('meta dir path', meta_dir_path)
#if that directory doesn't already exists create one
if not os.path.exists(meta_dir_path):
  os.mkdir(meta_dir_path)


groundplane_level = -2.0
# pc_sig  = 0.0002
# cluster_sig = 0.005
# pc_mean = 0.0
# cluster_mean = 0.05
# axis = 2  #z-axis
# sigmalist = np.array([pc_sig,cluster_sig])
# meanlist = np.array([pc_mean,cluster_mean])
# encoded_pc_pcd = pcl.PointCloud()

x = (0,1.4)
y = (0,1.2)
z = (0,1.5)

resolution = 0.025

if __name__ == '__main__':

  #initialize gloabals
  #settings.init()

  # ROS node initialization
  rospy.init_node('QIM_encode', anonymous = True)

  # Initialize color_list
  get_color_list.color_list = []

  # Traverse a given directory and work on each file
  # Directory of Kitti binary file

  data_directory = './data/test_data/'

  for filename in os.listdir(data_directory):
    if filename.endswith(".bin")  and filename.startswith("000014"):
    # if filename.endswith(".bin") :
      working_file_name = ntpath.basename(os.path.join(data_directory, filename)).split('.')[0]
      print('file currently working on %s'%(working_file_name) )

      encoded_pointcloud = []
      #****************************************************************************************************************************
                                                      # Read point cloud
      #****************************************************************************************************************************

      # read the point cloud from pcd file.. here the os.path.join(directory, filename) would be of type "./data/test_data/xxxx.bin"
      points_list = load_pc_from_bin(os.path.join(data_directory, filename))
      
      pc_camera_angle_filtered  = filter_camera_angle(points_list[:,:3])
      #clean_pc = pc_camera_angle_filtered.copy()
      pc_groundplane_filtered = filter_groundplane(np.copy(pc_camera_angle_filtered), groundplane_level)
      # pc_groundplane_filtered = pc_groundplane_filtered[70:90,:]
      # print('pc shae and first few values', pc_groundplane_filtered.shape, pc_groundplane_filtered)

      
      #****************************************************************************************************************************
                                                  # Encode the point cloud
      #****************************************************************************************************************************
      pc_input = pc_groundplane_filtered
      print('poitn cloud shape and values', pc_input.shape, pc_input)
      c = np.around(((pc_input - np.array([x[0],y[0],z[0]])) / resolution)).astype(np.int32)
      print('quantized input pc numpy representation', c)

      quantized_pc  = c*resolution + np.array([x[0],y[0],z[0]])
      print('quantized input pc values', quantized_pc) 

    #   encoded_quantized_values, encoded_CB = qim_quantize_restricted(pc_input)
    #   encoded_pc = np.array([encoded_quantized_values]).reshape(-1,3)
    #   print('encoded pc numpy representation', encoded_pc)

    #   encoded_quantized_pc  = getPointCloud_from_quantizedValues(encoded_pc, resolution, x,y,z)
    #   print('encoded pc values', encoded_quantized_pc) 
    # # d = ((pc_input - np.array([x[0],y[0],z[0]])) / resolution).astype(np.int32)

    #   # 3. Get the decoded codebook and quantized values of decoded point cloud   
    #   # Decode the point cloud and get the code book
    #   decoded_CB, decoded_quantized_values = qim_decode(encoded_quantized_pc)
    #   print('decoded pc numpy representation', np.array([decoded_quantized_values]).reshape(-1,3))
      
      # Encode the point cloud
      quantized_encoded_pointcloud, encoded_CB = qim_quantize_restricted( np.copy(pc_groundplane_filtered) )
      quantized_encoded_pointcloud_npy = np.array([quantized_encoded_pointcloud]).reshape(-1,3)
      print('quant encoded shape and size', quantized_encoded_pointcloud_npy.shape, quantized_encoded_pointcloud_npy)

      encoded_pointcloud = getPointCloud_from_quantizedValues(  np.copy(quantized_encoded_pointcloud_npy), resolution, x,y,z)
      print('encoded pc shape and first few values', encoded_pointcloud.shape, encoded_pointcloud)

      # encoded_pointcloud = add_gaussianNoise_clusters(np.copy(pc_camera_angle_filtered), cluster_corners_filtered.reshape(-1,8,3)[:], sorted_centeroids_indices[:], sigmalist, meanlist, axis)
      # print('encoded point cloud shape', encoded_pointcloud.shape)


      # #****************************************************************************************************************************
      #                                             #compute distortion coefficient
      # #****************************************************************************************************************************
      # pc_in = np.copy(pc_groundplane_filtered)
      # pc_out = np.copy(quantized_encoded_pointcloud_npy)
      # rmse = measure_distortion(pc_in, pc_out)
      # rospy.loginfo('distortion coefficient %s' %(rmse))
 
      # #****************************************************************************************************************************
      #                                             # Decode the point cloud
      # #****************************************************************************************************************************
      # # Decode the point cloud
      decoded_CB, decoded_quantized_values = qim_decode( np.copy(encoded_pointcloud) )
      
      decoded_codebook = np.array([decoded_CB]).reshape(-1,3)
      encoded_codebook = np.array([encoded_CB]).reshape(-1,3)
      
      # print('decoded_codebook', decoded_codebook)
      # print('encoded_codebook', encoded_codebook)
      
      compare_codebooks(encoded_codebook, decoded_codebook)


    else:
      continue 


  # clusters_publisher = rospy.Publisher("/pcl_clusters", PointCloud2, queue_size = 10000)
  i = 0
  
  # Spin while node is not shutdown
  while not rospy.is_shutdown():
  # while rospy.is_shutdown():
    # read pcl and the point cloud
    
    #raw point clouds
       
    if(len(encoded_pointcloud) !=0):
      publish_pc2(encoded_pointcloud, "/encode_encoded_pointcloud")
    else:
      rospy.logerr("%s in encode is empty", "encoded_pointcloud")

    if(len(pc_groundplane_filtered) !=0):
      publish_pc2(pc_groundplane_filtered, "/encode_groundfiltered_raw")
    else:
      rospy.logerr("%s in encode is empty", "pc_groundplane_filtered")
    
    
    print('spin count', i)
    i += 1
  rospy.spin()
