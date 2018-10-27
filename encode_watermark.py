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
#import settings
from helper_functions import *
from kitti_clustering import *
from tamper_pc import *


#   #Encoding steps:
#   1. read the point cloud
#   2. run it through filter_camera_angle_groundplane
#   3. get the clusters and their cluster_centeroids and filter the centeroids to get the relevant clusters
#   4. encode the z-axis of the point cloud (relevant clusters and remaining cloud) with different sigma gaussian noise and 

if __name__ == '__main__':

#initialize gloabals
  #settings.init()

 # ROS node initialization
  rospy.init_node('watermark_encodedecode', anonymous = True)

  # Initialize color_list
  get_color_list.color_list = []


#****************************************************************************************************************************
                                # STEP 1: read pcl and the point cloud
#****************************************************************************************************************************
#****************************************************************************************************************************
#****************************************************************************************************************************

  #velodyne_path = "./data/000002.bin"
  velodyne_path = "./data/000003.bin"
  #velodyne_path = "./data/000004.bin"
  #velodyne_path = "./data/000005.bin"
  #velodyne_path = "./data/000006.bin"
  # velodyne_path = "./data/002937.bin"
  # read the point cloud from pcd file
  points_list = load_pc_from_bin(velodyne_path)

  groundplane_level = -1.2

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
  clean_pc = pc_camera_angle_filtered.copy()
  pc_groundplane_filtered = filter_groundplane(pc_camera_angle_filtered[:], groundplane_level)

# STEP 3 : Get the clusters and their cluster_centeroids
  
  cluster_cloud_color, cluster_corners = kitti_cluster(pc_groundplane_filtered[:])
  # Here we get a flat (288,) element point array this could be reshaped 
  # to (12,8,3) i,e 12 corners with (8,3) or (96,3) i,e 96  x,y,z, points
  cluster_centeroids = get_clustercenteroid(cluster_corners[:])

  
  #print('cluster centeroid[:, 0]', cluster_centeroids[:, 0] )
  # These logical bounds which is a flat array of True or False of the size of number of clusters can be used
  # Extract corners from clusters that fall within the x,y range of interest
  cluster_logical_bounds = get_pc_logicaldivision(cluster_centeroids.reshape(-1,3)[:], x=(0,80), y=(-5, 5), z=(-1.5,1.5))
  #print('cluster logical bounds', cluster_logical_bounds)
  
  cluster_corners_filtered = cluster_corners.reshape(-1,8,3)[cluster_logical_bounds]
  #print('cluster corneres filtered shape', cluster_corners_filtered.shape)
  
  filtered_cluster_centeroids = apply_logicalbounds_topc(cluster_centeroids.reshape(-1,3), cluster_logical_bounds)
  #print('filtered centeroids shape & values', filtered_cluster_centeroids.shape, filtered_cluster_centeroids)
  print('filtered centeroids shape', filtered_cluster_centeroids.shape)
  
  #sort the cluster centeroids
  sorted_centeroids_indices  = get_sortedcenteroid_indices(filtered_cluster_centeroids[:])
  #print('**_**_****_**_****_**_****_**_****_**_****_**_****_**_****_**_****_**_****_**_****_**_**')
  #print('sorted centeroid indices', sorted_centeroids_indices)

  #Get the sorted list
  sorted_filtered_cluster_centeroids = get_sortedcenteroids(filtered_cluster_centeroids[:], sorted_centeroids_indices[:])
  #print('sorted centeroids', sorted_filtered_cluster_centeroids)

  #STEP 4: convert the cluster centeroids into kdtree
  #kdtree_centeroids = get_kdtree_ofpc(filtered_cluster_centeroids.astype(np.float32))
  
  #kdtree_centeroids = get_kdtree_ofpc(filtered_cluster_centeroids.astype(np.float32)) 

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
  
  pc_sig  = 0.0002
  cluster_sig = 0.005
  pc_mean = 0.0
  cluster_mean = 0.05
  axis = 2  #z-axis
  sigmalist = np.array([pc_sig,cluster_sig])
  meanlist = np.array([pc_mean,cluster_mean]) 
   
  # Encode the point cloud
  encoded_pointcloud = add_gaussianNoise_clusters(pc_camera_angle_filtered[:], cluster_corners_filtered.reshape(-1,8,3)[:], sorted_centeroids_indices[:], sigmalist, meanlist, axis)
  
  #cluster_to_show_num_encode  = sorted_centeroids_indices[2]
  cluster_to_show_num_encode = 2
  encoded_cluster_corners = get_clustercorner_totamper(cluster_corners_filtered.reshape(-1,8,3), cluster_to_show_num_encode)
  encoded_cluster_logicalbounds, x_dist, y_dist, z_dist, ym = get_cluster_logicalbound(pc_groundplane_filtered, encoded_cluster_corners)
  encoded_cluster = pc_groundplane_filtered[encoded_cluster_logicalbounds]
  

  #****************************************************************************************************************************
  #****************************************************************************************************************************


#****************************************************************************************************************************
                                            #tampering point cloud
#****************************************************************************************************************************
#****************************************************************************************************************************
#****************************************************************************************************************************
#****************************************************************************************************************************
#   #tampered_pc_objectaddition = pctempering_objectaddition(pc_camera_angle_filtered, clustertocopy)
#   #tampered_pc_objectremoval = pctempering_objectdeletion(pc_camera_angle_filtered, clustertocopy)

  #get copied cluster
  cluster_num = 0
  cluster_corner_totamper = get_clustercorner_totamper(cluster_corners_filtered.reshape(-1,8,3)[:], cluster_num)
  cluster_logicalbounds, x_dist, y_dist, z_dist, ym = get_cluster_logicalbound(clean_pc[:], cluster_corner_totamper[:])
#   #copy_cluster = get_clustercorner_totamper(cluster_corners_filtered.reshape(-1,8,3), cluster_logic_bound)
  
  #tempered_point_cloud =  pctampering_objectaddition(encoded_pointcloud[:], clean_pc[:], cluster_logicalbounds[:], x_dist, y_dist, z_dist, ym)
  # tempered_point_cloud = pctampering_objectdeletion(encoded_pointcloud[:], cluster_logicalbounds[:])
  tempered_point_cloud = pctampering_objectmove(encoded_pointcloud[:], clean_pc[:], cluster_logicalbounds[:], x_dist, y_dist, z_dist, ym)
  #tempered_point_cloud =  encoded_pointcloud
# #****************************************************************************************************************************
# #****************************************************************************************************************************


# #***************************************************************************************************************************
#                                                 #decoding section
# #****************************************************************************************************************************
# #****************************************************************************************************************************
# #****************************************************************************************************************************
# #****************************************************************************************************************************
  
  pc_groundplane_filtered_decode = filter_groundplane(tempered_point_cloud[:], groundplane_level)

  # STEP3 : Get the clusters and their cluster_centeroids

  cluster_cloud_color_decode, cluster_corners_decode = kitti_cluster(pc_groundplane_filtered_decode[:])
  # Here we get a flat (288,) element point array this could be reshaped 
  # to (12,8,3) i,e 12 corners with (8,3) or (96,3) i,e 96  x,y,z, points
  cluster_centeroids_decode = get_clustercenteroid(cluster_corners_decode)

  #print('cluster centeroid[:, 0]', cluster_centeroids[:, 0] )
  # These logical bounds which is a flat array of True or False of the size of number of clusters can be used
  # to extract corners from the cluster_corners_decode in the next step.
  cluster_logical_bounds_decode = get_pc_logicaldivision(cluster_centeroids_decode.reshape(-1,3)[:], x=(0,80), y=(-5, 5), z=(-1.5,1.5))
  #print('decode: cluster logical bounds', cluster_logical_bounds_decode)

  cluster_corners_filtered_decode = cluster_corners_decode.reshape(-1,8,3)[cluster_logical_bounds_decode]
  print('decode: cluster corneres filtered shape', cluster_corners_filtered_decode.shape)

  filtered_cluster_centeroids_decode = apply_logicalbounds_topc(cluster_centeroids_decode.reshape(-1,3)[:], cluster_logical_bounds_decode[:])
  #print('Decode: filtered centeroids decode shape & values', filtered_cluster_centeroids_decode.shape, filtered_cluster_centeroids_decode)
  print('Decode: filtered centeroids decode shape', filtered_cluster_centeroids_decode.shape)

  #sort the cluster centeroids
  sorted_centeroids_indices_decode  = get_sortedcenteroid_indices(filtered_cluster_centeroids_decode[:])
  #print('**_**_****_**_****_**_****_**_****_**_****_**_****_**_****_**_****_**_****_**_****_**_**')
  #print('sorted centeroid indices decode', sorted_centeroids_indices_decode)

  #Get the sorted list
  sorted_filtered_cluster_centeroids_decode = get_sortedcenteroids(filtered_cluster_centeroids_decode[:], sorted_centeroids_indices_decode[:])
  #print('sorted centeroids decode', sorted_filtered_cluster_centeroids_decode)


  threshold_distance =  0.5
  rcv_suspect_list, tx_suspect_list = get_clustercenteroid_changeindex(sorted_filtered_cluster_centeroids.reshape(-1,3)[:], sorted_filtered_cluster_centeroids_decode.reshape(-1,3)[:], threshold_distance)

  threshold_correlation = 0.1
  cluster_list_tampered = identify_tampered_clusters(rcv_suspect_list[:], tx_suspect_list[:], sorted_centeroids_indices[:], sorted_centeroids_indices_decode[:],  tempered_point_cloud[:], cluster_corners_filtered_decode[:], cluster_corners_filtered[:], threshold_correlation, sigmalist[:], meanlist[:], axis)

#****************************************************************************************************************************
#****************************************************************************************************************************



#****************************************************************************************************************************
                                            #other tests
#****************************************************************************************************************************
#****************************************************************************************************************************
#****************************************************************************************************************************

  # #prelim tests with index 2 and index 0

  # # get the correlation coefficients at all the indices of the clusters and figure out if it conicides with the above result
  # # Here: point_cloud is the cluster where you want to check the correlation, noise length is the length of the number of points in the given axis of given cluster 
  # #vairance is the initial variance of the watermarking encoding
 

  # # #cluster_culprit = get_clustercorners_totamper(cluster_corners_filtered_decode.reshape(-1,8,3), 0)
  # cluster_to_show_num_decode  = 2
  # culprit_cluster_corners = get_clustercorner_totamper(cluster_corners_filtered_decode.reshape(-1,8,3), cluster_to_show_num_decode)
  # culprit_cluster_logicalbounds, x_dist, y_dist, z_dist, ym = get_cluster_logicalbound(pc_groundplane_filtered_decode, culprit_cluster_corners)
  # culprit_cluster = pc_groundplane_filtered_decode[culprit_cluster_logicalbounds]
  # mean = 0
  # if(cluster_to_show_num_decode > 5 ):
  #   variance = 0.02
  # else:
  #   variance = sigmalist[sorted_centeroids_indices[2]]
  # axis = 2  
  # corr_value = linearcorrelation_comparison(mean, variance, culprit_cluster, axis) 

  # print ('*********^^^^^^^^****************corr value for cluster %s is %s' %(0, corr_value))
  
#****************************************************************************************************************************
#****************************************************************************************************************************

  i = 0
  # Spin while node is not shutdown
  while not rospy.is_shutdown():
    # read pcl and the point clid
    
    #raw point clouds
    publish_pc2(clean_pc, "/clean_pointcloud")
    publish_pc2(encoded_pointcloud, "/pc_camerafiltered")
    publish_pc2(pc_groundplane_filtered, "/pc_groundfiltered_raw")
    publish_pc2(pc_groundplane_filtered_decode, "/pc_groundfiltered_decode")
    
    publish_pc2(tempered_point_cloud, "/pc_tampered")

    #publish_pc2(encoded_pointcloud.reshape(-1,3)[:,:], "/pointcloud_watermarked")
    
    #clusters and centeroids encode
    #clusters_publisher.publish(cluster_cloud_color)
    #publish_pc2(cluster_corners.reshape(-1,3)[:,:], "/pointcloud_clustercorners")
    #publish_pc2(cluster_centeroids.reshape(-1,3)[:,:], "/pointcloud_clustercorners_centeroids")
    publish_pc2(sorted_filtered_cluster_centeroids.reshape(-1,3)[:,:], "/encode_centeroids_filtered")
    
    
    
  #clusters and centeroids decode
    publish_pc2(sorted_filtered_cluster_centeroids_decode.reshape(-1,3)[:,:], "/decode_centeroids_filtered")
    publish_pc2(cluster_list_tampered.reshape(-1,3)[:,:], "/tampered_clusters")
    #publish_pc2(culprit_cluster.reshape(-1,3)[:,:], "/culprit_cluster")
    publish_pc2(encoded_cluster.reshape(-1,3)[:,:], "/encoded_cluster")

    print('spinn count', i)
    i += 1
  rospy.spin()