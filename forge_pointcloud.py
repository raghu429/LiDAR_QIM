#!/usr/bin/env python
import sys
import numpy as np
import pcl
import rospy
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2, PointField
#from random import randint
import struct
import ctypes
import ntpath
import std_msgs.msg
from helper_functions import *
from kitti_clustering import *
from tamper_pc import *


#set the directory name for the encoded point clouds
forge_dir_path = os.path.join("./data", "forged")
#print('pc dir path', pc_dir_path)
#if that directory doesn't already exists create one
if not os.path.exists(forge_dir_path):
  os.mkdir(forge_dir_path)

#set the directory name for the encoded point clouds
add_dir_path = os.path.join("./data/forged", "added")
#print('pc dir path', pc_dir_path)
#if that directory doesn't already exists create one
if not os.path.exists(add_dir_path):
  os.mkdir(add_dir_path)

#set the directory name for the camera filtered point clouds
moved_pc_dir_path = os.path.join("./data/forged", "moved")
#print('pc dir path', pc_dir_path)
#if that directory doesn't already exists create one
if not os.path.exists(moved_pc_dir_path):
  os.mkdir(moved_pc_dir_path)

#set the directory name for meta data
del_dir_path = os.path.join("./data/forged", "deleted")
#print('meta dir path', meta_dir_path)
#if that directory doesn't already exists create one
if not os.path.exists(del_dir_path):
  os.mkdir(del_dir_path)

#set the directory name for verification tests
validate_dir_path = os.path.join("./data/forged", "validate")
#print('meta dir path', meta_dir_path)
#if that directory doesn't already exists create one
if not os.path.exists(validate_dir_path):
  os.mkdir(validate_dir_path)


#get copied cluster
cluster_num = 0

encoded_data_directory = './data/encoded/'
clean_data_directory = './data/camfiltered/'
metadata_directory = './data/metadata/'


if __name__ == '__main__':
    # ROS node initialization
    rospy.init_node('watermark_pcforge', anonymous = True)

    for filename in os.listdir(encoded_data_directory):
        if filename.endswith(".npy") and filename.startswith("000002"):
            encoded_pc = []
            cluster_source_pointcloud = []
            cluster_corners_filtered = []
            cluster_corner_totamper = []
            added_point_cloud = []
            deleted_point_cloud = []
            moved_point_cloud = []

            working_file_name = ntpath.basename(os.path.join(encoded_data_directory, filename)).split('.')[0] 
            print('file currently working on %s'%(working_file_name) )

            global_file_name  = working_file_name +'.npy'
            meta_file_name  = working_file_name + 'meta.npy'
            #get the cluster corners
            cluster_corners_filtered = np.load(os.path.join(metadata_directory, meta_file_name))

            #get the cluster source point cloud
            cluster_source_pointcloud = np.load(os.path.join(clean_data_directory, global_file_name))

            #get the encoded point cloud
            encoded_pc = np.load(os.path.join(encoded_data_directory, global_file_name))
            encoded_pc_copy = encoded_pc.copy()

            cluster_corner_totamper = get_clustercorner_totamper(cluster_corners_filtered.reshape(-1,8,3)[:], cluster_num)
            cluster_logicalbounds, x_dist, y_dist, z_dist, ym = get_cluster_logicalbound(cluster_source_pointcloud[:], cluster_corner_totamper[:])
            
            added_point_cloud, modified_cluster_center =  pctampering_objectaddition(np.copy(encoded_pc), cluster_source_pointcloud[:], cluster_logicalbounds[:], x_dist, y_dist, z_dist, ym)

            deleted_point_cloud = pctampering_objectdeletion(np.copy(encoded_pc), cluster_logicalbounds[:])

            moved_point_cloud, moved_cluster_center = pctampering_objectmove(np.copy(encoded_pc), cluster_source_pointcloud[:], cluster_logicalbounds[:], x_dist, y_dist, z_dist, ym)
            
            #get the cluster to tamper
            deleted_cluster_center = get_clustercenteroid(cluster_corner_totamper[:])

            #****************************************************************************************************************************
                                                        #Save the modified files
            #****************************************************************************************************************************
            #note when this file is read remember the notation that element 0 is moved centeroid, and element 1 is deleted centeriod
            np.save(os.path.join(validate_dir_path, global_file_name), (modified_cluster_center.reshape(-1,3), deleted_cluster_center.reshape(-1,3) ))
            
            np.save(os.path.join(add_dir_path, global_file_name), added_point_cloud)
            np.save(os.path.join(del_dir_path, global_file_name), deleted_point_cloud)
            np.save(os.path.join(moved_pc_dir_path, global_file_name), moved_point_cloud)

            break
        else:
            continue 

    i = 0

    # Spin while node is not shutdown
    while not rospy.is_shutdown():
        # read pcl and the point cloud
        
        #raw point clouds
        if(len(cluster_source_pointcloud) !=0):
            publish_pc2(cluster_source_pointcloud, "/forge_clean_pointcloud")
        else:
            rospy.logwarn("%s in forge code is empty", "cluster_source_pointcloud")
        
        if(len(encoded_pc) !=0):
            publish_pc2(encoded_pc, "/forge_encoded_pc")
        else:
            rospy.logerr("%s in forge code is empty", "encoded_pc")
        
        if(len(added_point_cloud) !=0):
            publish_pc2(added_point_cloud, "/forge_clusteradded_pc") 
        else:
            rospy.logerr("%s in forge code is empty", "added_point_cloud")
        
        if(len(deleted_point_cloud) !=0):
            publish_pc2(deleted_point_cloud, "/forge_clusterdeleted_pc")
        else:
            rospy.logerr("%s in forge code is empty", "deleted_point_cloud")
        
        if(len(moved_point_cloud) !=0):
            publish_pc2(moved_point_cloud, "/forge_clustermoved_pc")
        else:
            rospy.logerr("%s in forge code is empty", "moved_point_cloud")
            
        
        #publish_pc2(culprit_cluster.reshape(-1,3)[:,:], "/culprit_cluster")
        print('spin count', i)
        i += 1
    rospy.spin()













