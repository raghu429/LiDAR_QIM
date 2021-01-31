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


 
if __name__ == '__main__':
    # x = (0,1.4)
    # y = (0,1.2)
    # z = (0,1.5)

    # ROS node initialization
  rospy.init_node('QIM_VIZ', anonymous = True)

  clean_path = '/home/rchangs/work/SensorForensics/LIDAR/QIM/ACM_Feb2019/QIM_data/camfiltered/000071.npy'
  dither_path = '/home/rchangs/work/SensorForensics/LIDAR/QIM/ACM_Feb2019/QIM_data/encoded_Dither/bs4_dr2_000071.npy'
  plain_path = '/home/rchangs/work/SensorForensics/LIDAR/QIM/ACM_Feb2019/QIM_data/encoded_QIM/000071.npy'

  clean_pc = np.load(clean_path)
  normal_encoded_pc = np.load(plain_path)
  dither_encoded_pc = np.load(dither_path)


  i = 0

  # Spin while node is not shutdown
  while not rospy.is_shutdown():
  # while rospy.is_shutdown():
    # read pcl and the point cloud
    
    #raw point clouds
        
    if(len(normal_encoded_pc) !=0):
      publish_pc2(normal_encoded_pc, "/normal_qim_encoded_pointcloud")
    else:
      rospy.logerr("%s in encode is empty", "normal_qim_encoded_pointcloud")
    if(len(dither_encoded_pc) !=0):
      publish_pc2(dither_encoded_pc, "/dither_qim_encoded_pointcloud")
    else:
      rospy.logerr("%s in encode is empty", "dither_qim_encoded_pointcloud")    
    if(len(clean_pc) !=0):
      publish_pc2(clean_pc, "/clean_pointcloud")
    else:
      rospy.logerr("%s in encode is empty", "clean_pointcloud")

    
    
    print('spin count', i)
    i += 1
  rospy.spin()