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
import std_msgs.msg
from helper_functions import *
from kitti_clustering import *
from tamper_pc import *

#   #Encoding steps:
#   1. read the point cloud
#   2. run it through filter_camera_angle_groundplane
#   3. get the clusters and their cluster_centeroids and filter the centeroids to get the relevant clusters
#   4. encode the z-axis of the point cloud (relevant clusters and remaining cloud) with different sigma gaussian noise and 

if __name__ == '__main__':

 # ROS node initialization
  rospy.init_node('watermark_encodedecode', anonymous = True)

  # Initialize color_list
  get_color_list.color_list = []

# STEP 1: read pcl and the point cloud
#  velodyne_path = "./data/000002.bin"
#  velodyne_path = "./data/000003.bin"
#  velodyne_path = "./data/000004.bin"
#  velodyne_path = "./data/000005.bin"
#  velodyne_path = "./data/000006.bin"
  velodyne_path = "./data/002937.bin"
  # read the point cloud from pcd file
  points_list = load_pc_from_bin(velodyne_path)

# STEP 2: run it through filter_camera_angle_groundplane and remove gorund plane
  resolution = 0.2
  voxel_shape=(800, 800, 40)
  x=(0, 80)
  y=(-40, 40)
  z=(-2.5, 1.5)

  #logical_bounds = get_pc_logicaldivision(points_list, x=x, y=y, z=z)
  #pc_logicalbounds = apply_logicalbounds_topc(points_list, logical_bounds)
  #In the camera angle filter function we remove the ground  plane by doing z thresholding
  pc_camera_angle_filtered  = filter_camera_angle(points_list[:,:3])
  pc_groundplane_filtered = filter_groundplane(pc_camera_angle_filtered, -1.0)

# STEP 3 : Get the clusters and their cluster_centeroids
  
  cluster_cloud_color, cluster_corners = kitti_cluster(pc_groundplane_filtered)
  # Here we get a flat (288,) element point array this could be reshaped 
  # to (12,8,3) i,e 12 corners with (8,3) or (96,3) i,e 96  x,y,z, points
  cluster_centeroids = get_clustercenteroid(cluster_corners)
  
  #print('cluster centeroid[:, 0]', cluster_centeroids[:, 0] )
  # These logical bounds which is a flat array of True or False of the size of number of clusters can be used
  # Extract corners from clusters that fall within the x,y range of interest
  cluster_logical_bounds = get_pc_logicaldivision(cluster_centeroids.reshape(-1,3), x=(0,80), y=(-5, 5), z=(-1.5,1.5))
  print('cluster logical bounds', cluster_logical_bounds)
  
  cluster_corners_filtered = cluster_corners.reshape(-1,8,3)[cluster_logical_bounds]
  print('cluster corneres filtered shape', cluster_corners_filtered.shape)
  
  filtered_cluster_centeroids = apply_logicalbounds_topc(cluster_centeroids.reshape(-1,3), cluster_logical_bounds)
  print('filtered centeroids shape & values', filtered_cluster_centeroids.shape, filtered_cluster_centeroids)

  #STEP 4: convert the cluster centeroids into kdtree
  kdtree_centerids = get_kdtree_ofpc(filtered_cluster_centeroids.astype(np.float32)) 

  # STEP 4: Encode the z-axis of the point cloud (relevant clusters and remaining cloud) with different sigma gaussian noise
  
  #prepare the sigma and mean lists
  min_sigma = 0.1
  max_sigma  = 0.5
  gap = (max_sigma - min_sigma)/filtered_cluster_centeroids.shape[0]
  sigmalist = np.arange(min_sigma,max_sigma,gap)
  meanlist =  np.zeros(filtered_cluster_centeroids.shape[0])
  axis = 2
  
  #Bleow call seems to be a call by value hence copy and save the pc_camera_angle_filtered point cloud
  encoded_pc = pc_camera_angle_filtered
  
  # Encode the point cloud
  encoded_pointcoud = add_gaussianNoise_clusters(encoded_pc, cluster_corners_filtered.reshape(-1,8,3), sigmalist, meanlist, axis, 0.002, 0)
  #tampered_pc_objectaddition = pctempering_objectaddition(pc_camera_angle_filtered, clustertocopy)
  #tampered_pc_objectremoval = pctempering_objectdeletion(pc_camera_angle_filtered, clustertocopy)

  i = 0
  # Spin while node is not shutdown
  while not rospy.is_shutdown():
    # read pcl and the point clid
    
    #raw point clouds
    publish_pc2(pc_camera_angle_filtered, "/pc_camerafiltered")
    publish_pc2(pc_groundplane_filtered, "/pc_groundfiltered_raw")
    publish_pc2(encoded_pointcoud.reshape(-1,3)[:,:], "/pointcloud_watermarked")
    
    #clusters and centeroids
    #clusters_publisher.publish(cluster_cloud_color)
    publish_pc2(cluster_corners.reshape(-1,3)[:,:], "/pointcloud_clustercorners")
    publish_pc2(cluster_centeroids.reshape(-1,3)[:,:], "/pointcloud_clustercorners_centeroids")
    publish_pc2(filtered_cluster_centeroids.reshape(-1,3)[:,:], "/pointcloud_clustercorners_centeroids_filtered")
    # publish_pc2(cluster_corners_filtered.reshape(-1,3)[:,:], "/pointcloud_clustercorners_filtered")
    #publish_pc2(tampered_pc_objectaddition, "/tampered_addedobject")
    #publish_pc2(tampered_pc_objectremoval, "/tampered_removedobject")
    
    print('spinn count', i)
    i += 1
  rospy.spin()