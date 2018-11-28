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



encoded_data_directory = './data/encoded/'
clean_data_directory = './data/camfiltered/'
metadata_directory = './data/metadata/'

validate_dir_path = './data/forged/validate'
pc_deleted_dir_path = './data/forged/deleted'
pc_added_dir_path = './data/forged/added'
pc_moved_dir_path = './data/forged/moved'

file_counter = 0
localization_counter = 0
detection_counter = 0

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
    # Initialize color_list
    get_color_list.color_list = []


    # ROS node initialization
    rospy.init_node('watermark_decode', anonymous = True)

    glb_file_count = 0
    glb_detect_count = 0
    glb_loc_center_count = 0
    glb_loc_corner_count = 0

    #selected_dir = pc_deleted_dir_path
    # selected_dir = pc_added_dir_path
    selected_dir = pc_moved_dir_path

    for filename in os.listdir(selected_dir):
        #if filename.endswith(".npy") and filename.startswith("000005"):
        if filename.endswith(".npy"):
            working_file_name = ntpath.basename(os.path.join(pc_deleted_dir_path, filename)).split('.')[0] 
            print('file currently working on %s'%(working_file_name) )
            global_file_name  = working_file_name +'.npy'
            meta_file_name  = working_file_name + 'meta.npy'    
            #
            
            # read the input tampered point cloud
            tampered_pc = np.load(os.path.join(pc_deleted_dir_path, global_file_name))
            pc_groundplane_filtered_decode = filter_groundplane(np.copy(tampered_pc), groundplane_level)
            
            cluster_cloud_color_decode, cluster_corners_decode = kitti_cluster(np.copy(pc_groundplane_filtered_decode))
            
            if(len(cluster_corners_decode) != 0):
                # # Here we get a flat (288,) element point array this could be reshaped 
                # # to (12,8,3) i,e 12 corners with (8,3) or (96,3) i,e 96  x,y,z, points
                cluster_centeroids_decode = get_clustercenteroid(cluster_corners_decode)
                if(len(cluster_centeroids_decode) != 0):

                    #print('cluster centeroid[:, 0]', cluster_centeroids[:, 0] )
                    #These logical bounds which is a flat array of True or False of the size of number of clusters can be used
                    # # to extract corners from the cluster_corners_decode in the next step.
                    cluster_logical_bounds_decode = get_pc_logicaldivision(cluster_centeroids_decode.reshape(-1,3)[:], x=(0,80), y=(-5, 5), z=(-1.5,1.5))
                    #print('decode: cluster logical bounds', cluster_logical_bounds_decode)

                    cluster_corners_filtered_decode = cluster_corners_decode.reshape(-1,8,3)[cluster_logical_bounds_decode]
                    # print('decode: cluster corneres filtered shape', cluster_corners_filtered_decode.shape)
                    if(cluster_corners_filtered_decode.shape[0] != 0):
                        rospy.logwarn('number of the filtered clusters: %d', cluster_corners_filtered_decode.shape[0])
                        #get the validation centeroid list
                        #file is read remember the notation that element 0 is moved centeroid, and element 1 is deleted centeriod
                        validation_centeroid_list = np.load(os.path.join(validate_dir_path, global_file_name))

  
                        # #get the filtered cluster corners
                        cluster_corners_filtered = np.load(os.path.join(metadata_directory, meta_file_name))  
                        rospy.logwarn('number of the filtered clusters from source: %d', cluster_corners_filtered.shape[0])

                        #filtered cluster centeroids
                        filtered_cluster_centeroids = get_clustercenteroid(cluster_corners_filtered)

                        #sorted cluster centeroid indices
                        sorted_centeroids_indices  = get_sortedcenteroid_indices(np.copy(filtered_cluster_centeroids))

                        #sorted centroids (not indices)
                        sorted_filtered_cluster_centeroids = get_sortedcenteroids(filtered_cluster_centeroids[:], sorted_centeroids_indices[:])
                        print("***************************sorted_filtered_cluster_centeroids %s" %(sorted_filtered_cluster_centeroids))


                        filtered_cluster_centeroids_decode = apply_logicalbounds_topc(cluster_centeroids_decode.reshape(-1,3)[:], cluster_logical_bounds_decode[:])
                        # #print('Decode: filtered centeroids decode shape & values', filtered_cluster_centeroids_decode.shape, filtered_cluster_centeroids_decode)
                        # print('Decode: filtered centeroids decode shape', filtered_cluster_centeroids_decode.shape)

                        # #sort the cluster centeroids
                        sorted_centeroids_indices_decode  = get_sortedcenteroid_indices(filtered_cluster_centeroids_decode[:])
                        # #print('**_**_****_**_****_**_****_**_****_**_****_**_****_**_****_**_****_**_****_**_****_**_**')
                        # #print('sorted centeroid indices decode', sorted_centeroids_indices_decode)

                        # #Get the sorted list
                        sorted_filtered_cluster_centeroids_decode = get_sortedcenteroids(filtered_cluster_centeroids_decode[:], sorted_centeroids_indices_decode[:])
                        # #print('sorted centeroids decode', sorted_filtered_cluster_centeroids_decode)
                        print("***************************sorted_filtered_cluster_centeroids DECODE: %s" %(sorted_filtered_cluster_centeroids_decode))

                        glb_file_count += 1
                        threshold_distance =  0.5
                        rcv_suspect_list, tx_suspect_list = get_clustercenteroid_changeindex(sorted_filtered_cluster_centeroids.reshape(-1,3)[:], \
                        sorted_filtered_cluster_centeroids_decode.reshape(-1,3)[:], threshold_distance)

                        is_tampered = 1
                        threshold_correlation = 0.1
                        cluster_list_tampered, detect_count, loc_centercount, loc_cornercount = identify_tampered_clusters(is_tampered, validation_centeroid_list, \
                        rcv_suspect_list[:], tx_suspect_list[:], sorted_centeroids_indices[:],\
                        sorted_centeroids_indices_decode[:],  np.copy(tampered_pc), cluster_corners_filtered_decode[:], cluster_corners_filtered[:], \
                        threshold_correlation, sigmalist[:], meanlist[:], axis)
                        
                        glb_detect_count += detect_count
                        glb_loc_center_count  += loc_centercount
                        glb_loc_corner_count  += loc_cornercount

                        
                        # rospy.logwarn('selected delete folder')
                        # rospy.logwarn('Glb file count %d', glb_file_count)
                        # rospy.logwarn('detection count %d', glb_detect_count)
                        # rospy.logwarn('center detection count %d', glb_loc_center_count)



                        
    #                     #****************************************************************************************************************************
    #                                                     # Get the scores
    #                     #****************************************************************************************************************************

                    else:
                        rospy.logwarn("No decoded cluster corners found after filtering")
                else:
                    rospy.logwarn("No decoded cluster centeroids found")
            else:
                    rospy.logwarn("No decoded cluster corners found")
        else:
            continue


    rospy.logwarn('Glb file count %d', glb_file_count)
    rospy.logwarn('detection count %d', glb_detect_count)
    rospy.logwarn('center detection count %d', glb_loc_center_count)
    rospy.logwarn('corner detection count %d', glb_loc_corner_count)

    if(selected_dir == pc_deleted_dir_path):
        rospy.logwarn('selected delete folder')
        rospy.logerr("Detction accuracy: %f", ( ( glb_detect_count/(1.0*glb_file_count))*100 ) )
        rospy.logerr("localization accuracy: %f", ((glb_loc_center_count/(1.0*glb_file_count))*100 ) )
    elif (selected_dir == pc_added_dir_path):
        rospy.logwarn('selected add folder')
        rospy.logerr("Detction accuracy: %f", ((glb_detect_count/(1.0*glb_file_count))*100 ) )
        rospy.logerr("localization accuracy: %f", ((glb_loc_corner_count/(1.0*glb_file_count))*100 ))
    elif(selected_dir == pc_moved_dir_path):
        rospy.logwarn('selected move folder')
        rospy.logerr("Detction accuracy: %f", ((glb_detect_count/(1.0*glb_file_count))*100 ) )
        rospy.logerr("localization accuracy: %f", (( (glb_loc_center_count + glb_loc_corner_count) / (2.0*glb_file_count))*100))

    
    # # STEP3 : Get the clusters and their cluster_centeroids

    
    #****************************************************************************************************************************
    #****************************************************************************************************************************
    i = 0
    while not rospy.is_shutdown():
        
        if(len(tampered_pc) !=0):
            publish_pc2(tampered_pc, "/decode_pc_tampered")
        else:
            rospy.logerr("%s in decode code is empty", "tampered_pc")

        if(len(pc_groundplane_filtered_decode) !=0):
            publish_pc2(pc_groundplane_filtered_decode, "/decode_groundplane_filtered_pc")
        else:
            rospy.logerr("%s in decode code is empty", "pc_groundplane_filtered_decode")

        if(len(cluster_corners_decode) !=0):
            publish_pc2(cluster_corners_decode.reshape(-1,3)[:,:], "/decode_cluster_corners")
        else:
            rospy.logwarn("%s in encode is empty", "cluster_corners_decode")

        if(len(cluster_corners_filtered_decode) !=0):
            publish_pc2(cluster_corners_filtered_decode.reshape(-1,3)[:,:], "/decode_cluster_corners_filtered")
        else:
            rospy.logwarn("%s in encode is empty", "cluster_corners_filtered_decode")
    
        if(len(cluster_list_tampered) !=0):
            publish_pc2(cluster_list_tampered.reshape(-1,3)[:,:], "/decode_detected_clusters")
        else:
            rospy.logwarn("%s in encode is empty", "cluster_list_tampered")
    
        


        # print('spinn count', i)
        i += 1

        # publish_pc2(tampered_pc, "/pc_tampered")

        # #publish_pc2(encoded_pointcloud.reshape(-1,3)[:,:], "/pointcloud_watermarked")

        # #clusters and centeroids encode
        # clusters_publisher.publish(cluster_cloud_color)
        # #publish_pc2(cluster_corners.reshape(-1,3)[:,:], "/pointcloud_clustercorners")
        # #publish_pc2(cluster_centeroids.reshape(-1,3)[:,:], "/pointcloud_clustercorners_centeroids")
        # publish_pc2(sorted_filtered_cluster_centeroids.reshape(-1,3)[:,:], "/encode_centeroids_filtered")


        # if(len(sorted_filtered_cluster_centeroids_decode) !=0):
        # publish_pc2(sorted_filtered_cluster_centeroids_decode.reshape(-1,3)[:,:], "/decode_centeroids_filtered")
        # else:
        # rospy.logerr("%s is empty", sorted_filtered_cluster_centeroids_decode)
        # #clusters and centeroids decode

        # if(len(cluster_list_tampered) !=0):
        # publish_pc2(cluster_list_tampered.reshape(-1,3)[:,:], "/tampered_clusters")
        # else:
        # rospy.logerr("%s is empty", "cluster_list_tampered")
    rospy.spin()