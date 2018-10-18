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

def pctempering_objectaddition(pointcloud_input, cluster_to_copy):
    pc = pointcloud_input.reshape(-1,3)
    #define the boundary of the cluster
    x_min,y_min,z_min,x_max,y_max,z_max =  get_min_max_ofpc(cluster_to_copy)
    #define the location to copy the data.. like moving the entire cluster to the left (add ymax-yim+0.5 to the ycomponents of points)
     
    #get the logical bounds
    clusterlogical_bound = get_pc_logicaldivision(pc, x=(x_min, x_max), y=(y_min, y_max), z=(z_min, z_max))
    print('cluter logic bound shape', clusterlogical_bound.shape, clusterlogical_bound)
    #modify the point cloud with the noise
    #get the cluster
    cluster_to_copy = pc[clusterlogical_bound]
    #get the displacement
    y_range = y_max -y_min
    displacemnet = y_range*1.5
   
    #modify the cluster
    cluster_to_copy[:,1] = cluster_to_copy[:,1] + displacemnet

    #concatenate the cluster with point cloud
    updated_point_cloud = np.concatenate((pc, cluster_to_copy), axis =0)

    print('****************************************************************************************************')
    
    print('clusterotcopy hspae and values', cluster_to_copy.shape, cluster_to_copy)
    print('pc shpae and values', pc[clusterlogical_bound].shape, pc[clusterlogical_bound])

    return updated_point_cloud

def pctempering_objectdeletion(pointcloud_input, cluster_to_copy):
    pc = pointcloud_input.reshape(-1,3)
    #define the boundary of the cluster
    x_min,y_min,z_min,x_max,y_max,z_max =  get_min_max_ofpc(cluster_to_copy)
    #define the location to copy the data.. like moving the entire cluster to the left (add ymax-yim+0.5 to the ycomponents of points)
     
    #get the logical bounds
    clusterlogical_bound = get_pc_logicaldivision(pc, x=(x_min, x_max), y=(y_min, y_max), z=(z_min, z_max))
    print('cluter logic bound shape', clusterlogical_bound.shape, clusterlogical_bound)
    #modify the point cloud with the noise
    #get the cluster
    cluster_to_copy = pc[clusterlogical_bound]
    
    #modify the cluster
    dummy = np.zeros(pc[clusterlogical_bound].shape)

    #modify the point cloud with the noise
    pc[clusterlogical_bound] = dummy

    return pc

def get_noiseaddedcloud(pointcloud_input, logical_bound_in, sigma_in, mean_in, axis_in):
#Make a zeros matrix with same size as the logical_bounds
    dummy = np.zeros(pointcloud_input[logical_bound_in].shape)
    #make gaussian noise to the z-axis of this dummy matrix
    noise = np.random.normal(mean_in, sigma_in, dummy[:,axis_in].shape[0])
    #Add gaussian noise to zero matrix
    dummy[:,axis_in] = dummy[:,axis_in] + noise
    #modify the point cloud with the noise
    pointcloud_input[logical_bound_in] = pointcloud_input[logical_bound_in] + dummy


def get_cluster_totamper(pointcloud_raw, cluster_list, cluster_num):
    pc = pointcloud_raw.reshape(-1,3)
    #super_logicalbound = np.zeros(pc.shape[0])
    #super_logicalbound_not = np.zeros(pc.shape[0])
    if(cluster_num < cluster_list.shape[0]):
        (x_min,y_min,z_min,x_max,y_max,z_max) = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        cluster = cluster_list[i].reshape(-1,3)
        print('cluster shapes', cluster.shape)
        copied_cluster = cluster
    else:
        print('******************------------------------*************')    
        print('cluster requested doesnt exist')

    return copied_cluster


#Have the raw camera filtered point cloud (-1,3) format and runt throhg the clustering
#       #for each cluster corner in cluster_corners_filtered - get the min,max of each axis to populate the x,y,z bounds - (x,) format
#       #Then run the get_pc_logicalbounds and apply_logicalbounds_topc to highlight the cluster points
#take the z axis and apply gaussian noise to the z components of these points
#for the remaining cloud add a different gaussian noise

def add_gaussianNoise_clusters(pointcloud_raw, cluster_list, sigmalist, mean_list, axis, global_sigma, global_mean):
    pc = pointcloud_raw.reshape(-1,3)
    print('******************------------------------*************')
    super_logicalbound = np.zeros(pc.shape[0])
    super_logicalbound_not = np.zeros(pc.shape[0])
    for i in range(0, cluster_list.shape[0]):
        (x_min,y_min,z_min,x_max,y_max,z_max) = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        cluster = cluster_list[i].reshape(-1,3)
        print('-*-*-*-*')
        print('cluster shapes', cluster.shape)
        #get the boundaries of the cluster
        x_min,y_min,z_min,x_max,y_max,z_max =  get_min_max_ofpc(cluster)
        #get the logical bounds
        clusterlogical_bound = get_pc_logicaldivision(pc, x=(x_min, x_max), y=(y_min, y_max), z=(z_min, z_max))
        #add the logical bound to global bound
        super_logicalbound =  np.logical_or(super_logicalbound,clusterlogical_bound)
        #add noise 
        get_noiseaddedcloud(pc, clusterlogical_bound, sigmalist[i], mean_list[i], axis)

    #modify the z-axis of remaining pointcloud with a fixed noise param
    super_logicalbound_not =  np.logical_not(super_logicalbound)
    get_noiseaddedcloud(pc, super_logicalbound_not, global_sigma, global_mean, axis)

    #finally return the modified pointcloud
    return pc
    