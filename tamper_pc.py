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

def get_cluster_logicalbound(pointcloud_input, clustercorner_to_copy):
    pc = pointcloud_input.reshape(-1,3)
    #define the boundary of the cluster
    x_min,y_min,z_min,x_max,y_max,z_max =  get_min_max_ofpc(clustercorner_to_copy)
    y_range = y_max -y_min
    x_range = x_max -x_min
    z_range = z_max -z_min

    #define the location to copy the data.. like moving the entire cluster to the left (add ymax-yim+0.5 to the ycomponents of points)
        
    #get the logical bounds
    clusterlogical_bound = get_pc_logicaldivision(pc, x=(x_min, x_max), y=(y_min, y_max), z=(z_min, z_max))
    # print('cluter logic bound shape and value', clusterlogical_bound.shape, clusterlogical_bound)
    # print('cluter logic bound shape', clusterlogical_bound.shape)
    #modify the point cloud with the noise
    #get the cluster
    #cluster_to_copy = pc[clusterlogical_bound]

    return clusterlogical_bound, x_range, y_range, z_range, y_max


def pctampering_objectaddition(pointcloud_input, clean_pc, logical_bound, xrange, yrange, zrange, ymax):
    pc = pointcloud_input.reshape(-1,3)
    
    #get the cluster to be replicated
    cluster_to_copy = clean_pc[logical_bound]
    
    #get the displacement
    
    displacemnet = yrange*2
   
    if (ymax < 0):
        #modify the cluster
        cluster_to_copy[:,1] = cluster_to_copy[:,1] + displacemnet
    else:
        #modify the cluster
        cluster_to_copy[:,1] = cluster_to_copy[:,1] - displacemnet
        
    cluster_center = np.array([np.average(cluster_to_copy[:,0]), np.average(cluster_to_copy[:,1]), np.average(cluster_to_copy[:,2])])
    #concatenate the cluster with point cloud
    updated_point_cloud = np.concatenate((pc, cluster_to_copy), axis =0)
    
    #print('clusterotcopy shape and values', cluster_to_copy.shape, cluster_to_copy)
    print('cluster to copy shape', cluster_to_copy.shape)
    #print('pc shpae and values', pc[clusterlogical_bound].shape, pc[clusterlogical_bound])

    return updated_point_cloud, cluster_center


def pctampering_objectdeletion(pointcloud_input, logical_bound):
    pc = pointcloud_input.reshape(-1,3)
    #modify the cluster
    #dummy = np.zeros(pc[logical_bound].shape)
    #modify the point cloud with the noise
    # cluster_to_copy = pc[logical_bound]
    # cluster_to_copy[:,2] = 0

    # pc[logical_bound][:][:,:] = 0
    pc[logical_bound] = 0
    # print('pc logical bound', pc[logical_bound][0:10])
    # print('pc logical bound x', pc[logical_bound][: ][:,0])
    # pc[logical_bound] = 0

    return pc

def pctampering_objectmove(pointcloud_input, clean_pc, logical_bound, xrange, yrange, zrange, ymax):
    pc = pctampering_objectdeletion(pointcloud_input,logical_bound)
    pc_out, cluster_centeroid = pctampering_objectaddition(pc[:], clean_pc, logical_bound, xrange, yrange, zrange, ymax)

    return pc_out, cluster_centeroid


def get_noiseaddedcloud(pointcloud_input, logical_bound_in, sigma_in, mean_in, axis_in):
    point_cloud  = []
#Make a zeros matrix with same size as the logical_bounds
    dummy = np.zeros(pointcloud_input[logical_bound_in].shape)
    #make gaussian noise to the z-axis of this dummy matrix
    noise = np.random.normal(mean_in, sigma_in, dummy[:,axis_in].shape[0])
    print('mean, sigma, length inside function******', mean_in, sigma_in, dummy[:,axis_in].shape[0])
    #Add gaussian noise to zero matrix
    dummy[:,axis_in] = dummy[:,axis_in] + noise
    #modify the point cloud with the noise
    pointcloud_input[logical_bound_in] = pointcloud_input[logical_bound_in] + dummy
    # point_cloud = pointcloud_input[logical_bound_in]
    # print('point cloud shae from function', point_cloud.shape)
    # #correlation heck within the function
    # corr_val = linearcorrelation_comparison(mean_in, sigma_in, point_cloud, axis_in)
    # print('corr value from function', corr_val) 
    # print('\n')


def linearcorrelation_comparison(mean, variance, point_cloud, axis):
    correlation_value = 0.0
    noise_length = point_cloud[:, axis].shape[0]
    #print('\n')
    print('noise length in correlation', noise_length)
    reference_noise = np.random.normal(mean, variance, noise_length)
    # Do the correlation
    lcs = np.correlate(point_cloud[:, axis], reference_noise)
    correlation_value = (lcs/noise_length) 
    #correlation_value = linear_correlation(point_cloud[:, axis], reference_noise)

    #print('lcs', correlation_value)
    return (correlation_value)


def get_clustercorner_totamper(cluster_list, cluster_num):
    if(cluster_num < cluster_list.shape[0]):
        (x_min,y_min,z_min,x_max,y_max,z_max) = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        cluster = cluster_list[cluster_num].reshape(-1,3)
        #print('cluster shapes', cluster.shape)
        copied_cluster_corner = cluster
    else:
        print('******************------------------------*************')    
        print('cluster requested doesnt exist')
        copied_cluster_corner = []

    return copied_cluster_corner


#Have the raw camera filtered point cloud (-1,3) format and runt throhg the clustering
#       #for each cluster corner in cluster_corners_filtered - get the min,max of each axis to populate the x,y,z bounds - (x,) format
#       #Then run the get_pc_logicalbounds and apply_logicalbounds_topc to highlight the cluster points
#take the z axis and apply gaussian noise to the z components of these points
#for the remaining cloud add a different gaussian noise

def add_gaussianNoise_clusters(pointcloud_raw, cluster_list, sorted_index_list, sigmalist, mean_list, axis):
    pc_local = pointcloud_raw.reshape(-1,3)
    #print('******************------------------------*************')
    super_logicalbound = np.zeros(pc_local.shape[0])
    super_logicalbound_not = np.zeros(pc_local.shape[0])
    print('\n')
    
    sigma_pc  = sigmalist[0]
    sigma_cluster = sigmalist[1]
    mean_pc = mean_list[0]
    mean_cluster = mean_list[1]

    print('sigma list', sigmalist)
    print('mean list list', mean_list)
    print('sorted index list', sorted_index_list)

    for i in range(0, cluster_list.shape[0]):
        (x_min,y_min,z_min,x_max,y_max,z_max) = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        cluster = cluster_list[i].reshape(-1,3)
        print('-*-*-*-*')
        print('cluster shapes', cluster.shape)
        #get the boundaries of the cluster
        x_min,y_min,z_min,x_max,y_max,z_max =  get_min_max_ofpc(cluster[:])
        #get the logical bounds
        clusterlogical_bound = get_pc_logicaldivision(pc_local[:], x=(x_min, x_max), y=(y_min, y_max), z=(z_min, z_max))
        #add the logical bound to global bound
        super_logicalbound =  np.logical_or(super_logicalbound[:],clusterlogical_bound[:])
        #add noise 

        #cluster_initvaluescluster_initvalues = pc_local[clusterlogical_bound]
        #print('init cluster z values and length', cluster_initvalues.shape, cluster_initvalues[:, axis]) 
        print('\n')
        print('lklklklkkkklllllllllllllllllllllllllllllllllkkkkkkkkkkkkkkkkkkkkk')
        #get_noiseaddedcloud(pc_local, clusterlogical_bound, sigmalist[sorted_index_list[i]], mean_list[sorted_index_list[i]], axis)
        get_noiseaddedcloud(pc_local, clusterlogical_bound, sigma_cluster, mean_cluster, axis)
        #check the correlation to
        
        culprit_cluster = pointcloud_raw[clusterlogical_bound]
        #print('final cluster z values and length', culprit_cluster.shape, culprit_cluster[:, axis])
        
        #print('&&&&&&&& ************ original cluster index', i)

        # for j in range(0, len(mean_list)):
        #     corr_value = 0.0
        #     print('sigma & mean list of cluster %s is %s %s' %(j,sigmalist[sorted_index_list[j]], mean_list[sorted_index_list[j]]))
        #     corr_value = linearcorrelation_comparison(mean_list[sorted_index_list[j]], sigmalist[sorted_index_list[j]], culprit_cluster, axis) 
        #     print ('!*!* for loop: %s corr value: %s' %(j,corr_value))

        corr_value_pc = linearcorrelation_comparison(mean_pc, sigma_pc, culprit_cluster[:], axis) 
        print ('mean: %s, sigma:%s, corr value pc: %s' %(mean_pc, sigma_pc, corr_value_pc))

        corr_value_cluster = linearcorrelation_comparison(mean_cluster, sigma_cluster, culprit_cluster[:], axis) 
        print ('mean: %s, sigma:%s, corr value cluster: %s' %(mean_cluster, sigma_cluster, corr_value_cluster))

        if(abs(corr_value_pc) < abs(corr_value_cluster)):
            print('HOHOHOHOHOHOHOHHOHOH ___________ cluster is distinct from pc by %s amount' %(abs(corr_value_cluster) - abs(corr_value_pc)))
            print('\n')        
        else:
            print('bokka ayyindi')
            print('\n')

    #modify the z-axis of remaining pointcloud with a fixed noise param
    super_logicalbound_not =  np.logical_not(super_logicalbound)
    get_noiseaddedcloud(pc_local, super_logicalbound_not, sigma_pc, mean_pc, axis)

    #finally return the modified pointcloud
    return pc_local


def get_clustercenteroid(cluster_corner_list):
    #make sure that the shape is what you expect index * rows (8) * columns (3)
    cluster_corner_list = np.reshape(cluster_corner_list,(-1, 8, 3))
    print('cluster_corner_list shape', cluster_corner_list.shape)
    cluster_centeroid_list = np.array([[np.average(cluster_corner_list[i,:,0]), np.average(cluster_corner_list[i,:,1]), np.average(cluster_corner_list[i,:,2])] for i in range(0, cluster_corner_list.shape[0])])
    
    #print ('cluster centeroid shape and value', cluster_centeroid_list.shape, cluster_centeroid_list)
    print ('cluster centeroid shape', cluster_centeroid_list.shape)
    #print('\n')

    return cluster_centeroid_list

def get_kdtree_ofpc(points):
    #In this function since we try to get a kd-tree from point cloud library we need to convert the input numpy array into point cloud
    pc_points = pcl.PointCloud(points)
    kdtree_points = pc_points.make_kdtree_flann()

    return(kdtree_points)
  

def linear_correlation(a,b):
    #compare the sizes
    a_size = len(a)
    b_size = len(b)

    #print('a size', a_size)
    #print('a:', a)
    #print('b size', b_size)
    #print('b:', b)

    if(a_size != b_size):
        raise Exception('inputs have different size')
    #initialize the correlation
    lin_corr = 0.0
    for i in range(a_size):
        lin_corr += float(a[i])*float(b[i])
    
    lin_corr = float(lin_corr/(a_size))
    #print('linear corr value from function', lin_corr)
    return(lin_corr)


