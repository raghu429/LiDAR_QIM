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


groundplane_level = -1.2
pc_sig  = 0.0002
cluster_sig = 0.005
pc_mean = 0.0
cluster_mean = 0.05
axis = 2  #z-axis
sigmalist = np.array([pc_sig,cluster_sig])
meanlist = np.array([pc_mean,cluster_mean])
encoded_pc_pcd = pcl.PointCloud()

if __name__ == '__main__':

  #initialize gloabals
  #settings.init()

  # ROS node initialization
  rospy.init_node('watermark_encodedecode', anonymous = True)

  # Initialize color_list
  get_color_list.color_list = []

  # Traverse a given directory and work on each file
  # Directory of Kitti binary file

  data_directory = './data/test_data/'

  for filename in os.listdir(data_directory):
    if filename.endswith(".bin")  and filename.startswith("000012"):
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

      #****************************************************************************************************************************
                                                      # clustering
      #****************************************************************************************************************************
      cluster_cloud_color, cluster_corners = kitti_cluster(np.copy(pc_groundplane_filtered))
      # Here we get a flat (288,) element point array this could be reshaped 
      # to (12,8,3) i,e 12 corners with (8,3) or (96,3) i,e 96  x,y,z, points
      if(len(cluster_corners) != 0):
        cluster_centeroids = get_clustercenteroid(cluster_corners[:])
        #print('cluster centeroids', cluster_centeroids)
        rospy.loginfo('cluster centeroids length: %s', len(cluster_centeroids))

        if(len(cluster_centeroids) != 0):

          #print('cluster centeroid[:, 0]', cluster_centeroids[:, 0] )
          # These logical bounds which is a flat array of True or False of the size of number of clusters can be used
          # Extract corners from clusters that fall within the x,y range of interest
          cluster_logical_bounds = get_pc_logicaldivision(cluster_centeroids.reshape(-1,3)[:], x=(0,80), y=(-5, 5), z=(-1.5,1.5))
          #print('cluster logical bounds', cluster_logical_bounds)
          cluster_corners_filtered = cluster_corners.reshape(-1,8,3)[cluster_logical_bounds]
          rospy.loginfo('cluster corners filtered shape: %s', cluster_corners_filtered.shape)
          
          if(cluster_corners_filtered.shape[0] != 0):
            #****************************************************************************************************************************
                                                        #Display cluster for debug purposes
            #****************************************************************************************************************************
            #display a cluster 
            cluster_to_show_num_encode = 0
            encoded_cluster_corners = get_clustercorner_totamper(cluster_corners_filtered.reshape(-1,8,3), cluster_to_show_num_encode)
            encoded_cluster_logicalbounds, x_dist, y_dist, z_dist, ym = get_cluster_logicalbound(pc_groundplane_filtered[:], encoded_cluster_corners[:])
            encoded_cluster = pc_groundplane_filtered[encoded_cluster_logicalbounds]
            
            #****************************************************************************************************************************
                                                        # Sort and filter the cluster centeroids
            #****************************************************************************************************************************
            #we could also get the centeroids by calling get_clustercenteroid on the filtered corner list
            filtered_cluster_centeroids = apply_logicalbounds_topc(cluster_centeroids.reshape(-1,3), cluster_logical_bounds)
            #print('filtered centeroids shape & values', filtered_cluster_centeroids.shape, filtered_cluster_centeroids)
            rospy.loginfo('filtered centeroids shape: %s', filtered_cluster_centeroids.shape)

            #sort the cluster centeroids
            sorted_centeroids_indices  = get_sortedcenteroid_indices(filtered_cluster_centeroids[:])
            #print('sorted centeroid indices', sorted_centeroids_indices)

            #Get the sorted list
            sorted_filtered_cluster_centeroids = get_sortedcenteroids(filtered_cluster_centeroids[:], sorted_centeroids_indices[:])
            #print('sorted centeroids', sorted_filtered_cluster_centeroids)

            #****************************************************************************************************************************
                                                        # Encode the point cloud
            #****************************************************************************************************************************
            # Encode the point cloud
            encoded_pointcloud = add_gaussianNoise_clusters(np.copy(pc_camera_angle_filtered), cluster_corners_filtered.reshape(-1,8,3)[:], sorted_centeroids_indices[:], sigmalist, meanlist, axis)
            print('encoded point cloud shape', encoded_pointcloud.shape)


            #****************************************************************************************************************************
                                                        #compute distortion coefficient
            #****************************************************************************************************************************
            pc_in = np.copy(pc_camera_angle_filtered)
            pc_out = np.copy(encoded_pointcloud)
            rmse = measure_distortion(pc_in, pc_out)
            rospy.loginfo('distortion coefficient %s' %(rmse))
            
            #****************************************************************************************************************************
                                                        #Save the point cloud and the meta data
            #****************************************************************************************************************************

            # #save the encoded point cloud as pcd file
            # encoded_pc_pcd = pcl.PointCloud
            # encoded_pc_pcd.from_array(np.array(list(encoded_pointcloud), dtype= np.float32))

            working_file_name_pcd = working_file_name +'.npy'
            #save encoded pint cloud
            np.save(os.path.join(pc_dir_path, working_file_name_pcd), encoded_pointcloud)
            #save corresponding clean point cloud
            np.save(os.path.join(clean_pc_dir_path, working_file_name_pcd), pc_camera_angle_filtered)

            #save the metadata to a txt file  
            working_file_name_txt = working_file_name + 'meta'+ '.npy'
            np.save(os.path.join(meta_dir_path, working_file_name_txt), cluster_corners_filtered.reshape(-1,8,3)[:])            

          else:
              rospy.logwarn('cluster centeroids filtered has zero length')  
        else:
            rospy.logwarn('cluster centeroids has zero length')
      else:
        rospy.logwarn('No clusters found')
    else:
      continue 


  clusters_publisher = rospy.Publisher("/pcl_clusters", PointCloud2, queue_size = 10000)
  i = 0
  
  # Spin while node is not shutdown
  while not rospy.is_shutdown():
    # read pcl and the point cloud
    
    #raw point clouds


    if(len(cluster_corners) !=0):
      publish_pc2(cluster_corners.reshape(-1,3), "/encode_cluster_corners")
    else:
      rospy.logwarn("%s in encode is empty", "cluster_corners")
    
    if(len(cluster_corners_filtered) !=0):
      publish_pc2(cluster_corners_filtered.reshape(-1,3), "/encode_cluster_corners_filtered")
    else:
      rospy.logwarn("%s in encode is empty", "cluster_corners_filtered")
    


    if(len(pc_camera_angle_filtered) !=0):
      publish_pc2(pc_camera_angle_filtered, "/encode_clean_pointcloud")
    else:
      rospy.logwarn("%s in encode is empty", "pc_camera_angle_filtered")
      
    if(len(encoded_pointcloud) !=0):
      publish_pc2(encoded_pointcloud, "/encode_encoded_pointcloud")
    else:
      rospy.logerr("%s in encode is empty", "encoded_pointcloud")

    if(len(pc_groundplane_filtered) !=0):
      publish_pc2(pc_groundplane_filtered, "/encode_groundfiltered_raw")
    else:
      rospy.logerr("%s in encode is empty", "pc_groundplane_filtered")
    
    if(len(encoded_cluster) !=0):
      publish_pc2(encoded_cluster.reshape(-1,3)[:,:], "/encode_encoded_cluster")
    else:
      rospy.logerr("%s in encode is empty", "encoded_cluster")
    
    #publish_pc2(culprit_cluster.reshape(-1,3)[:,:], "/culprit_cluster")
    print('spin count', i)
    i += 1
  rospy.spin()


#***************************************************************************************************************************************

  # STEP 2: run it through filter_camera_angle_groundplane and remove gorund plane
  # resolution = 0.2
  # voxel_shape=(800, 800, 40)
  # x=(0, 80)
  # y=(-40, 40)
  # z=(-2.5, 1.5)

  #logical_bounds = get_pc_logicaldivision(points_list, x=x, y=y, z=z)
  #pc_logicalbounds = apply_logicalbounds_topc(points_list, logical_bounds)
  #In the camera angle filter function we remove the ground  plane by doing z thresholding
  

# STEP 3 : Get the clusters and their cluster_centeroids
      
    # STEP 4: Encode the z-axis of the point cloud (relevant clusters and remaining cloud) with different sigma gaussian noise  
    #prepare the sigma and mean lists
    #min_sigma = 0.001
    #max_sigma  = 0.001
    #gap = (max_sigma - min_sigma)/filtered_cluster_centeroids.shape[0]
    #sigmalist = np.array([ min_sigma + min_sigma*(i **(3)) for i in range(filtered_cluster_centeroids.shape[0]) ])
    #sigmalist = np.arange(min_sigma,max_sigma,gap)
    
    #min_mean = 0.0
    #max_mean  = 0.5
    #gap = (max_mean - min_mean)/filtered_cluster_centeroids.shape[0]
    #meanlist = np.arange(min_mean,max_mean,gap)
    #meanlist =  np.zeros(filtered_cluster_centeroids.shape[0])
  
  #   #Encoding steps:
#   1. read the point cloud
#   2. run it through filter_camera_angle_groundplane
#   3. get the clusters and their cluster_centeroids and filter the centeroids to get the relevant clusters
#   4. encode the z-axis of the point cloud (relevant clusters and remaining cloud) with different sigma gaussian noise and 


# pc_dir_path = os.path.join('data', 'encoded')
# print('pc dir path', pc_dir_path)
# #if that directory doesn't already exists create one
# if not os.path.exists(os.path.dirname(pc_dir_path)):
#   try:
#       pc_folder = os.mkdir(pc_dir_path)
#   except OSError as exc: # Guard against race condition
#       if exc.errno != errno.EEXIST:
#           raise
# else:
#   pc_folder = pc_dir_path
#   print('encoded pcd files are stored at', pc_folder)

# #set the directory name for meta data
# meta_dir_path = os.path.join('data', 'metadata')
# print('meta dir path', meta_dir_path)
# #if that directory doesn't already exists create one
# if not os.path.exists(os.path.dirname(meta_dir_path)):
#   try:
#       print('path doesnt exist')
#       metadata_folder = os.mkdir(meta_dir_path)
#   except OSError as exc: # Guard against race condition
#       if exc.errno != errno.EEXIST:
#           raise
# else:
#   metadata_folder = meta_dir_path
#   print('metadata files are stored at', metadata_folder)

#****************************************************************************************************************************
                                # STEP 1: read pcl and the point cloud
#****************************************************************************************************************************
#****************************************************************************************************************************
#****************************************************************************************************************************

  
  
  
  
  
  
  
  
  
  
  
  
  
  
  #velodyne_path = "./data/test_data/000002.bin"
  # velodyne_path = "./data/000003.bin"
  # velodyne_path = "./data/000004.bin"
  #velodyne_path = "./data/000005.bin"
  #velodyne_path = "./data/000006.bin"
  #velodyne_path = "./data/002937.bin"




  # Total 30 test files
  # velodyne_path = "./data/test_data/000002.bin"
  # velodyne_path = "./data/test_data/000003.bin"
  # velodyne_path = "./data/test_data/000004.bin"
  # velodyne_path = "./data/test_data/000006.bin"
  # velodyne_path = "./data/test_data/000007.bin"
  # velodyne_path = "./data/test_data/000010.bin"

  #velodyne_path = "./data/test_data/000011.bin"
  # velodyne_path = "./data/test_data/000012.bin"
  # velodyne_path = "./data/test_data/000018.bin"
  # velodyne_path = "./data/test_data/000019.bin"
  # velodyne_path = "./data/test_data/000026.bin"

  # velodyne_path = "./data/test_data/000027.bin"
  velodyne_path = "./data/test_data/000028.bin"
  # velodyne_path = "./data/test_data/000030.bin"
  # velodyne_path = "./data/test_data/000033.bin"
  # velodyne_path = "./data/test_data/000034.bin"

  # velodyne_path = "./data/test_data/000035.bin"
  # velodyne_path = "./data/test_data/000038.bin"
  # velodyne_path = "./data/test_data/000039.bin"
  # velodyne_path = "./data/test_data/000043.bin"
  # velodyne_path = "./data/test_data/000044.bin"

  # velodyne_path = "./data/test_data/000047.bin"
  # velodyne_path = "./data/test_data/000048.bin"
  # velodyne_path = "./data/test_data/000051.bin"
  # velodyne_path = "./data/test_data/000054.bin"
  # velodyne_path = "./data/test_data/000055.bin"

  # velodyne_path = "./data/test_data/000060.bin"
  # velodyne_path = "./data/test_data/000061.bin"
  # velodyne_path = "./data/test_data/000064.bin"
  # velodyne_path = "./data/test_data/000067.bin"
  # velodyne_path = "./data/test_data/000074.bin"
# set the output file directory