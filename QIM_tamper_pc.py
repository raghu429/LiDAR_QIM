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
from tamper_pc import *
from QIM_helper import *



#set the directory name for the modified point clouds
forge_dir_path = os.path.join("./QIM_data", "forged")
#print('pc dir path', pc_dir_path)
#if that directory doesn't already exists create one
if not os.path.exists(forge_dir_path):
  os.mkdir(forge_dir_path)

# #set the directory name for the encoded point clouds
# add_dir_path = os.path.join("./QIM_data/forged", "added")
# #print('pc dir path', pc_dir_path)
# #if that directory doesn't already exists create one
# if not os.path.exists(add_dir_path):
#   os.mkdir(add_dir_path)

# #set the directory name for the camera filtered point clouds
# moved_pc_dir_path = os.path.join("./QIM_data/forged", "moved")
# #print('pc dir path', pc_dir_path)
# #if that directory doesn't already exists create one
# if not os.path.exists(moved_pc_dir_path):
#   os.mkdir(moved_pc_dir_path)

# #set the directory name for meta data
# del_dir_path = os.path.join("./QIM_data/forged", "deleted")
# #print('meta dir path', meta_dir_path)
# #if that directory doesn't already exists create one
# if not os.path.exists(del_dir_path):
#   os.mkdir(del_dir_path)

# #set the directory name for verification tests
# validate_dir_path = os.path.join("./QIM_data/forged", "validate")
# #print('meta dir path', meta_dir_path)
# #if that directory doesn't already exists create one
# if not os.path.exists(validate_dir_path):
#   os.mkdir(validate_dir_path)




encoded_data_directory = './QIM_data/encoded/'
clean_data_directory = './QIM_data/camfiltered/'
label_dir = './QIM_data/label_2'
calib_dir = './QIM_data/calib'


nonoise_clean_dir_path = './QIM_data/forged/nonoise/clean/'
nonoise_deleted_dir_path = './QIM_data/forged/nonoise/deleted/'
nonoise_added_dir_path = './QIM_data/forged/nonoise/added/'
# nonoise_moved_dir_path = './QIM_data/forged/nonoise/moved/'

lownoise_clean_dir_path = './QIM_data/forged/lownoise/clean/'
lownoise_deleted_dir_path = './QIM_data/forged/lownoise/deleted/'
lownoise_added_dir_path = './QIM_data/forged/lownoise/added/'
# lownoise_moved_dir_path = './QIM_data/forged/lownoise/moved/'


highnoise_clean_dir_path = './QIM_data/forged/highnoise/clean/'
highnoise_deleted_dir_path = './QIM_data/forged/highnoise/deleted/'
highnoise_added_dir_path = './QIM_data/forged/highnoise/added/'
# highnoise_moved_dir_path = './QIM_data/forged/highnoise/moved/'



# resolution_halfdelta = resolution/2.0
resolution = 0.05
threshold_sigma = resolution/6.93 #sigma < stepsize/4*sqrt(3) 

# low_sigma = threshold_sigma - 0.003
low_sigma = 0.0025
high_sigma = 0.004

# high_sigma = threshold_sigma + 0.001
mu = 0.0

if __name__ == '__main__':
    # ROS node initialization
    rospy.init_node('watermark_pcforge', anonymous = True)

    for filename in os.listdir(encoded_data_directory):
        # encoded_pc = []
        # clean_source_pc = []
        # # cluster_corners_filtered = []
        # cluster_corner_totamper = []
        # tampered_pc_addition = []
        # deleted_point_cloud = []
        # moved_point_cloud = []
        # tampered_points =[] 

        if filename.endswith(".npy") and filename.startswith("002937"):
        # if filename.endswith(".npy"):
            encoded_pc = []
            clean_source_pc = []
            # cluster_corners_filtered = []
            cluster_corner_totamper = []
            tampered_pc_addition = []
            deleted_pc = []
            moved_point_cloud = []
            tampered_points =[] 

            working_file_name = ntpath.basename(os.path.join(encoded_data_directory, filename)).split('.')[0] 
            print('file currently working on %s'%(working_file_name) )

            global_file_name  = working_file_name +'.npy'
            # meta_file_name  = working_file_name + 'meta.npy'
            label_file_name = working_file_name + '.txt'
            calib_file_name = working_file_name + '.txt'

            #get the cluster source point cloud
            clean_source_pc = np.load(os.path.join(clean_data_directory, global_file_name))

            #get the encoded point cloud
            encoded_pc = np.load(os.path.join(encoded_data_directory, global_file_name))

            #****************************************************************************************************************************
                                                        #Get the corners of object to add/substract/move
            #****************************************************************************************************************************

            #read label and get the gt_bounding box corners
            places, rotation, sizes = read_labels(os.path.join(label_dir, label_file_name), 'txt', calib_path = None , is_velo_cam=False, proj_velo=None)
            
            if places is None:
                print('no labels found')
                continue
            else:
                gt_corners = visualize_groundtruth(os.path.join(calib_dir,calib_file_name), os.path.join(label_dir, label_file_name))
                # print('corners', gt_corners.shape, gt_corners)

                if(gt_corners.shape[0]):
                    #pick a label .. here we are  picking label 0 .. just to be on the same side
                    logical_bound, x_range, y_range, z_range, y_max = get_cluster_logicalbound (np.copy(clean_source_pc), gt_corners[0])
                    
                    # a = np.array([np.where(logical_bound == True)])
                    # print('logical bound', a.shape, a)
                else:
                    print("GT label list is empty")
                    continue
                
            #****************************************************************************************************************************
                                                        #Tampering: Addition
            #****************************************************************************************************************************
            tampered_pc_addition, tampered_cluster_center = pctampering_objectaddition(np.copy(encoded_pc), np.copy(encoded_pc), np.copy(logical_bound), x_range, y_range, z_range, y_max)

            # print("encoded pc size", encoded_pc.shape)
            # print("Added pc size", tampered_pc_addition.shape)
            
            
            #****************************************************************************************************************************
                                                        #Tampering: Deletion
            #****************************************************************************************************************************
            
            deleted_pc = pctampering_objectdeletion(np.copy(encoded_pc), np.copy(logical_bound))
            
            #****************************************************************************************************************************


            np.save(os.path.join(nonoise_added_dir_path, global_file_name), tampered_pc_addition)
            np.save(os.path.join(nonoise_deleted_dir_path, global_file_name), deleted_pc)
            np.save(os.path.join(nonoise_clean_dir_path, global_file_name), encoded_pc)

            print("encoded pc size", tampered_pc_addition.shape)
            # print("deleted pc size", deleted_pc.shape)
        
            
            #****************************************************************************************************************************
                                        #Tampering: Low Noise Addition
            #****************************************************************************************************************************
            # random_noise  = np.random.normal(0, 0.2, 3*len(tampered_pc_addition)).reshape(-1,3)

            # print('shape of random noise', random_noise.shape)
            # low_sigma = 0.001

            lownoise_tampered_pc_addition =  tampered_pc_addition + np.random.normal(mu, low_sigma, 3*len(tampered_pc_addition)).reshape(-1,3)

            lownoise_deleted_pc = deleted_pc + np.random.normal(mu, low_sigma, 3*len(deleted_pc)).reshape(-1,3)

            lownoise_clean_source_pc = encoded_pc + np.random.normal(mu, low_sigma, 3*len(encoded_pc)).reshape(-1,3)

            np.save(os.path.join(lownoise_added_dir_path, global_file_name), lownoise_tampered_pc_addition)
            np.save(os.path.join(lownoise_deleted_dir_path, global_file_name), lownoise_deleted_pc)
            np.save(os.path.join(lownoise_clean_dir_path, global_file_name), lownoise_clean_source_pc)



            #****************************************************************************************************************************
                                        #Tampering: High Noise Addition
            #****************************************************************************************************************************
            
        
            highnoise_tampered_pc_addition =  tampered_pc_addition + np.random.normal(mu, high_sigma, 3*len(tampered_pc_addition)).reshape(-1,3)

            highnoise_deleted_pc = deleted_pc + np.random.normal(mu, high_sigma, 3*len(deleted_pc)).reshape(-1,3)

            highnoise_clean_source_pc = encoded_pc + np.random.normal(mu, high_sigma, 3*len(encoded_pc)).reshape(-1,3)
            
            np.save(os.path.join(highnoise_added_dir_path, global_file_name), highnoise_tampered_pc_addition)
            np.save(os.path.join(highnoise_deleted_dir_path, global_file_name), highnoise_deleted_pc)
            np.save(os.path.join(highnoise_clean_dir_path, global_file_name), highnoise_clean_source_pc)

    
            #****************************************************************************************************************************
                                                        # Decode the point cloud
            #****************************************************************************************************************************
            # # Decode the point cloud
            resolution = 0.05
            resolution_halfdelta = resolution/2.0
            # decoded_CB, decoded_quantized_values = qim_decode( np.copy(deleted_pc), resolution_halfdelta )

            decoded_CB, decoded_quantized_values = qim_decode( np.copy(tampered_pc_addition), resolution_halfdelta )
            

            # decoded_CB, decoded_quantized_values = qim_decode( np.copy(deleted_pc), resolution_halfdelta )
            
            # decoded_CB, decoded_quantized_values = qim_decode( np.copy(tampered_pc_addition), resolution_halfdelta )
            


            decoded_codebook = np.array([decoded_CB]).reshape(-1,3)
            # encoded_codebook = np.array([encoded_CB]).reshape(-1,3)
            
            # print('decoded_codebook', decoded_codebook)
            # print('encoded_codebook', encoded_codebook)
            
            # compare_codebooks(encoded_codebook, decoded_codebook)
            # embedded_bits = 3
            
            tampered_indices, er_rate = get_tamperedindices_sequential_threebits(decoded_codebook)
            print('length of suspect indices', tampered_indices.shape[0], tampered_indices)
            # print('tampered pc', tampered_indices[0][1])

            tampered_points = tampered_pc_addition[tampered_indices]
            # tampered_points = encoded_pc[tampered_indices]


        else:
            continue 

    i = 0

    # Spin while node is not shutdown
    while not rospy.is_shutdown():
        # read pcl and the point cloud
        
        #raw point clouds
        if(len(clean_source_pc) !=0):
            publish_pc2(clean_source_pc, "/qim_tamper_clean_pc")
        else:
            rospy.logwarn("%s in forge code is empty", "clean_source_pc")
        
        if(len(encoded_pc) !=0):
            publish_pc2(encoded_pc, "/qim_tamper_encoded_pc")
        else:
            rospy.logerr("%s in forge code is empty", "encoded_pc")
        
        
        if(len(tampered_points) !=0):
            publish_pc2(tampered_points, "/qim_tamper_decoded_pc") 
        else:
            rospy.logerr("%s in forge code is empty", "tampered_points")
        
        if(len(tampered_pc_addition) !=0):
            publish_pc2(tampered_pc_addition, "/qim_tamper_tampered_pc_addition") 
        else:
            rospy.logerr("%s in forge code is empty", "tampered_pc_addition")
        
        if(len(deleted_pc) !=0):
            publish_pc2(deleted_pc, "/forge_clusterdeleted_pc")
        else:
            rospy.logerr("%s in forge code is empty", "deleted_point_cloud")
        
        # if(len(moved_point_cloud) !=0):
        #     publish_pc2(moved_point_cloud, "/forge_clustermoved_pc")
        # else:
        #     rospy.logerr("%s in forge code is empty", "moved_point_cloud")
    
        
        #publish_pc2(culprit_cluster.reshape(-1,3)[:,:], "/culprit_cluster")
        print('spin count', i)
        i += 1
    rospy.spin()













