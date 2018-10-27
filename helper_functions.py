#!/usr/bin/env python
import sys
import os
import rospy
import numpy as np
import cv2
import pcl
import glob
import math
import std_msgs.msg
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2

def load_pc_from_pcd(pcd_path):
    """Load PointCloud data from pcd file."""
    p = pcl.load(pcd_path)
    return np.array(list(p), dtype=np.float32)

def load_pc_from_bin(bin_path):
    """Load PointCloud data from pcd file."""
    obj = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    return obj

def publish_pc2(pc, topic):
    """Publisher of PointCloud data"""
    pub1 = rospy.Publisher(topic, PointCloud2, queue_size=10000)
    #rospy.init_node("pc2_publisher")
    header = std_msgs.msg.Header()
    header.stamp = rospy.Time.now()
    header.frame_id = "velodyne"
    points = pc2.create_cloud_xyz32(header, pc[:, :3])
    pub1.publish(points)
    
def get_min_max_ofpc(pc):
    x_min = np.min(pc[:,0]).astype(np.float32)
    y_min = np.min(pc[:,1]).astype(np.float32)
    z_min = np.min(pc[:,2]).astype(np.float32)
    
    x_max = np.max(pc[:,0]).astype(np.float32)
    y_max = np.max(pc[:,1]).astype(np.float32)
    z_max = np.max(pc[:,2]).astype(np.float32)
    
    print('min_x: %s, min_y: %s, min_z: %s, max_x: %s, max_y: %s, max_z: %s' %(x_min,y_min,z_min, x_max,y_max,z_max))
    return x_min,y_min,z_min, x_max,y_max,z_max

def dist(x,y):   
    return np.sqrt(np.sum((x-y)**2))

def get_pc_logicaldivision(pc, x=(0, 90), y=(-50, 50), z=(-4.5, 5.5)):
    """Convert PointCloud2 to Voxel"""

    #print('max and min values of raw point cloud')
    get_min_max_ofpc(pc)

    #note: the original point cloud has four columns
    print('pc shape before any modificatins', pc.shape)
    #print('pc before modification first 25 elements', pc[:25,:])

    #we start the filtering process to filter out the point cloud that we dont need
    logic_x = np.logical_and(pc[:, 0] >= x[0], pc[:, 0] < x[1])
    #print('logic_x length', np.count_nonzero(logic_x == 1))
    
    logic_y = np.logical_and(pc[:, 1] >= y[0], pc[:, 1] < y[1])
    #print(logic_y)
    #print('logic_y length', np.count_nonzero(logic_y == 1))

    logic_z = np.logical_and(pc[:, 2] >= z[0], pc[:, 2] < z[1])
    #print('logic_z length', np.count_nonzero(logic_z == 1))


    #print(logic_z)
    #print(np.logical_and(logic_x, np.logical_and(logic_y, logic_z)))
    #print('before pc', pc[:,:3])
    
    #apply the logical map to the point cloud (all rows and first three columns)
    #this would select all the rows in pc that had True value in the np.logical_and(logic_x, np.logical_and(logic_y, logic_z))
    #this part finishes the filtering of unwanted point cloud by selecting only the rows that are relevant
    logic_yz = np.logical_and(logic_y, logic_z)
    #print('logic_yz length', np.count_nonzero(logic_yz == 1))

    logic_x_yz = np.logical_and(logic_x, logic_yz)
    #print('logic_yz length', np.count_nonzero(logic_x_yz == 1))

    return logic_x_yz

def apply_logicalbounds_topc(pc, logical_division):

    pc_logical_division = pc[:, :3][logical_division]
    
    print('max and min values of pc after logical distribution')
    get_min_max_ofpc(pc_logical_division)

    print('after logical anding pc shape', pc_logical_division.shape)
    #print('after logical anding first 25 elements', pc_logical_division[:25,:])
    
    return pc_logical_division


def quantize_pc(pc, resolution=0.50, x=(0, 90), y=(-50, 50), z=(-4.5, 5.5), quant = 'low'):
    
    # Substract the min values from the point cloud to get the range and then divide it into resolution number for sets
    #and type cast to integer to get the discrete points with specified resolution
    if (quant == 'low'):
        pc = ((pc - np.array([x[0], y[0], z[0]])) / resolution).astype(np.int32)
    elif (quant == 'high'):
        pc = np.ceil((pc - np.array([x[0], y[0], z[0]])) / resolution).astype(np.int32)
    elif (quant == 'voxgrid'): #To Do: implement the voxel grid functionality
        pc = ((pc - np.array([x[0], y[0], z[0]])) / resolution).astype(np.int32)
    else: # implement low as default
        pc = ((pc - np.array([x[0], y[0], z[0]])) / resolution).astype(np.int32)


    print('pc shape after quantization:', pc.shape)
    print('pc after quantization first 25 elements', pc[:25, :])

    return pc

def draw_to_voxel(pc, resolution=0.50, x=(0, 90), y=(-50, 50), z=(-4.5, 5.5)):

    # # make a voxel with give shape length of x,y,x/resolution 
    # #here for example you have a voxel shape of 800, 800, 40 the with a granularity of 1, should have 800*800*40 points..
    # #but we dont have some many points so we only lightup the points that correspond to the point cloud.. 
    
    voxel = np.zeros((int((x[1] - x[0]) / resolution), int((y[1] - y[0]) / resolution), int(round((z[1]-z[0]) / resolution))))
    #print('insdie the drawvoxel function')
    print('voxel shape defined', voxel.shape)
    #print ('voxel defined first hundred ', voxel[:-100,:2,:])

    # #This is the step where you assign a value of 1 to the voxel grid at all the x,y,z points in the pc. The range of indices fall within the range of voxel
    # #size because of the quantization step. say pc is m X 3, then pc[:,0] would be an array of size m and so are the pc[:,1] and pc[:,2]. But the range of these m values would be within the range of the 
    # #voxel shape for ex.. if the voxel shape is (4,5,6) then then range of values of m would be below 6 due to the quantization step. So we are saying at all these points make the value = 1.

    #here we light up the points in the voxel that correspond to the point cloud
    voxel[pc[:, 0], pc[:, 1], pc[:, 2]] = 1
    
    print('voxel of pc elements shape', voxel.shape)
    #print('voxel of pc elements first 100', voxel[:-100, :2, :])
    
    return voxel

def filter_camera_angle(places):
    """Filter camera angles for KiTTI Datasets"""
    print('places point cloud shape', places.shape)
    xy_Camerafilter = np.logical_and((places[:, 1] < places[:, 0] - 0.27), (-places[:, 1] < places[:, 0] - 0.27))
    print('xy filter length', np.count_nonzero(xy_Camerafilter == 1))
    # z_filter = (places[:, 2] > -1.0)
    # print('z filter length', np.count_nonzero(z_filter == 1))
    # bool_in = np.logical_and(xy_filter, z_filter)
    # print('filter camera shape', bool_in.shape)
    # print('filter camera with Ture values', np.count_nonzero(bool_in == 1))
    # #bool_in = np.logical_and((places[:, 1] < places[:, 0]), (-places[:, 1] < places[:, 0]))
    return places[xy_Camerafilter]

def filter_groundplane(places, z_threshold):
    """Filter camera angles for KiTTI Datasets"""
    #print('places point cloud shape', places.shape)
    #xy_filter = np.logical_and((places[:, 1] < places[:, 0] - 0.27), (-places[:, 1] < places[:, 0] - 0.27))
    #print('xy filter length', np.count_nonzero(xy_filter == 1))
    z_filter = (places[:, 2] > z_threshold)
    print('z filter length', np.count_nonzero(z_filter == 1))
    #bool_in = np.logical_and(xy_filter, z_filter)
    #print('filter camera shape', bool_in.shape)
    #print('filter camera with Ture values', np.count_nonzero(bool_in == 1))
    #bool_in = np.logical_and((places[:, 1] < places[:, 0]), (-places[:, 1] < places[:, 0]))
    return places[z_filter]



def filter_camera_angle(places):
    """Filter camera angles for KiTTI Datasets"""
    bool_in = np.logical_and((places[:, 1] < places[:, 0] - 0.27), (-places[:, 1] < places[:, 0] - 0.27))
    print('filter camera shape', bool_in.shape)
    #bool_in = np.logical_and((places[:, 1] < places[:, 0]), (-places[:, 1] < places[:, 0]))
    return places[bool_in]

# Returns Downsampled version of a point cloud
# The bigger the leaf size the less information retained

def do_voxel_grid_filter(point_cloud, LEAF_SIZE = 0.01):
  voxel_filter = point_cloud.make_voxel_grid_filter()
  voxel_filter.set_leaf_size(LEAF_SIZE, LEAF_SIZE, LEAF_SIZE) 
  return voxel_filter.filter()