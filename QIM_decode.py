#!/usr/bin/env python
import sys
import numpy as np
# np.set_printoptions(threshold=np.nan)
import pcl
import rospy
import ntpath
import sensor_msgs.point_cloud2 as pc2
import shutil
from sensor_msgs.msg import PointCloud2, PointField
from random import randint
#import settings
from helper_functions import *
from tamper_pc import *
from QIM_helper import *



encoded_data_directory = './QIM_data/encoded/'
clean_data_directory = './QIM_data/camfiltered/'
label_dir = './QIM_data/label_2/'
calib_dir = './QIM_data/calib/'
img_dir = './QIM_data/test_img/'
unwanted_img_dir = './QIM_data/test_img_unwanted/'


nonoise_clean_dir_path = './QIM_data/forged/nonoise/clean/'
nonoise_deleted_dir_path = './QIM_data/forged/nonoise/deleted/'
nonoise_added_dir_path = './QIM_data/forged/nonoise/added/'
nonoise_moved_dir_path = './QIM_data/forged/nonoise/moved/'

lownoise_clean_dir_path = './QIM_data/forged/lownoise/clean/'
lownoise_deleted_dir_path = './QIM_data/forged/lownoise/deleted/'
lownoise_added_dir_path = './QIM_data/forged/lownoise/added/'
lownoise_moved_dir_path = './QIM_data/forged/lownoise/moved/'


highnoise_clean_dir_path = './QIM_data/forged/highnoise/clean/'
highnoise_deleted_dir_path = './QIM_data/forged/highnoise/deleted/'
highnoise_added_dir_path = './QIM_data/forged/highnoise/added/'
highnoise_moved_dir_path = './QIM_data/forged/highnoise/moved/'



resolution = 0.05
resolution_halfdelta = resolution/2.0
label_tomove_index = 0
# threshold_sigma = resolution/6.93 #sigma < stepsize/4*sqrt(3) 
# low_sigma = threshold_sigma - 0.01
# high_sigma = threshold_sigma + 0.01
# mu = 0.0

def move_gt_label(cluster_to_copy,yrange, ymax):
    #get the cluster to be replicated
    displacemnet = yrange*2
   
    if (ymax < 0):
        #modify the cluster
        cluster_to_copy[:,1] = cluster_to_copy[:,1] + displacemnet
    else:
        #modify the cluster
        cluster_to_copy[:,1] = cluster_to_copy[:,1] - displacemnet
        
    return cluster_to_copy

if __name__ == '__main__':

    # ROS node initialization
    rospy.init_node('watermark_decode', anonymous = True)

    # dir_list = [nonoise_deleted_dir_path, nonoise_added_dir_path, nonoise_clean_dir_path] 
    
    dir_list = [lownoise_deleted_dir_path, lownoise_added_dir_path, lownoise_clean_dir_path]
    
    # dir_list = [ highnoise_deleted_dir_path, highnoise_added_dir_path, highnoise_clean_dir_path]
    
        

    for index, dir_name in enumerate(dir_list):
        selected_dir = dir_name
        print('\n')
        # print('-------------------new dir: Dist------------', selected_dir)
        print('-------------------new dir: BER------------', selected_dir)
        print('dir count:', index)
        print('\n')
        # print('new file:', glb_detect_count)
        glb_file_count = 0
        glb_FN_count = 0
        glb_FP_count = 0
     
        for filename in os.listdir(selected_dir):
            tampered_pc = []
            clean_source_pc = []
            tampered_points = []
            generated_boundingbox = []
            gt_boundingbox = []

            
            # if filename.endswith(".npy") and filename.startswith("001179"):
            if filename.endswith(".npy"):
                working_file_name = ntpath.basename(os.path.join(selected_dir, filename)).split('.')[0] 

                # print('\n')
                # # print('\n')
                # print('file currently working on %s'%(working_file_name) )
                global_file_name  = working_file_name +'.npy'
                # meta_file_name  = working_file_name + 'meta.npy'    
                
                #get the cluster source point cloud
                clean_source_pc = np.load(os.path.join(clean_data_directory, global_file_name))
                
                # read the input tampered point cloud
                # print('loading file: from ', global_file_name, os.path.join(selected_dir, global_file_name))
                tampered_pc = np.load(os.path.join(selected_dir, global_file_name))
                # print('loaded file size', tampered_pc.size)

                decoded_CB, decoded_quantized_values = qim_decode( np.copy(tampered_pc), resolution_halfdelta )
                
                decoded_codebook = np.array([decoded_CB]).reshape(-1,3)

                tampered_indices, b_errorRate = get_tamperedindices_sequential_threebits(decoded_codebook)

                # tampered_indices = np.array([tampered_indices])
                
                # print('length of suspect indices', tampered_indices.shape[0], tampered_indices)
                # print('suspect indice length', tampered_indices.shape[0])

                if(len(tampered_indices) != 0): #check to see we got an intact point cloud
                    
                    if('clean' in selected_dir):
                        glb_FP_count += 1

                    if('delete' in selected_dir ):
                        # read the reference input cloud. Since the tampered pc wont have any points at these locations, if we extract points from it, we'll get no corners.
                        reference_pc = np.load(os.path.join(encoded_data_directory, global_file_name))
                        tampered_points = reference_pc[tampered_indices]
                    else:
                        tampered_points = tampered_pc[tampered_indices]

                    
                    tampered_points =  np.array(tampered_points)
                    # print("tampered points", tampered_points.shape, tampered_points)
                    # print("tampered points", tampered_points.shape)
                    
                    #****************************************************************************************************************************
                                    #detection accuracy IoU
                    #****************************************************************************************************************************
                    #GT bounding box 
                    label_file_name = working_file_name + '.txt'
                    calib_file_name = working_file_name + '.txt'

                    #read label and get the gt_bounding box corners
                    places, rotation, sizes = read_labels(os.path.join(label_dir, label_file_name), 'txt', calib_path = None , is_velo_cam=False, proj_velo=None)
                
                    if(places is None):
                        print('no labels found')
                        # shutil.move(img_dir+working_file_name+'.png', unwanted_img_dir)
                        continue

                    gt_corners = visualize_groundtruth(os.path.join(calib_dir,calib_file_name), os.path.join(label_dir, label_file_name))
                    # print('corners', gt_corners.shape, gt_corners)

                    if(gt_corners.shape[0]):
                        #pick a label .. here we are  picking label 0 .. just to be on the same side
                        logical_bound, x_range, y_range, z_range, y_max = get_cluster_logicalbound (np.copy(clean_source_pc), gt_corners[label_tomove_index])
                        a = np.array([np.where(logical_bound == True)])
                        # print('logical bound shape', a.shape)
                    else:
                        print("GT label list is empty")
                        continue
                        
                    #generate bounding box based on shifting the co-ordinates from original label (this is like ground truth)
                    if('add' in selected_dir ):
                        gt_boundingbox = move_gt_label(np.copy(gt_corners[label_tomove_index]), y_range, y_max)
                    else:
                        gt_boundingbox = gt_corners[label_tomove_index]


                    gt_boundingbox = gt_boundingbox.reshape(-1,3)
                    # gt_boundingbox = np.array([bb_a[0], bb_a[1], bb_a[2], bb_a[3]])
                
                    #Generated bounding box
                    generated_boundingbox = get_boundingboxcorners(tampered_points)
                    # print('generated bb', generated_boundingbox)
                    generated_boundingbox = generated_boundingbox.reshape(-1,3)
                    # print("bb_b shape", bb_b.shape)
                    # generated_boundingbox = np.array([bb_b[0], bb_b[1], bb_b[2], bb_b[5]])

                    #calculate IoU
                    # overlap_area = bb_intersection_over_union(generated_boundingbox[:,:2], gt_boundingbox[:,:2])
                    # print("IOU", overlap_area)
                    distortion = Hausdorff_dist(generated_boundingbox, gt_boundingbox )
                    # if(distortion > 0.25):
                    #     # shutil.move(img_dir+working_file_name+'.png', unwanted_img_dir)
                    #     print('**** Distortion high file name:', working_file_name)
                    
                else:
                    # print('no tampered indices found')
                    if('clean' not in selected_dir):
                        glb_FN_count += 1
                        # shutil.move(img_dir+working_file_name+'.png', unwanted_img_dir)
                    # else:
                
                glb_file_count+=1

                # 
                print b_errorRate
                # print distortion


                # print('global file count:', glb_file_count)
                
                # print ('FN:', glb_FN_count)
                # print ('FP:', glb_FP_count)
                # print('localization accuracy:', distortion)
                # print('BER:', b_errorRate)
                        

            else:
                print('no file found')
            
            

        print('global file count:', glb_file_count)
        print ('FN:', glb_FN_count)
        print ('FP:', glb_FP_count)
        # break

    i = 0
    print ('***********************************')
    
    while not rospy.is_shutdown():

        if(len(clean_source_pc) !=0):
            publish_pc2(clean_source_pc, "/qim_tamper_clean_pc")
        else:
            rospy.logwarn("%s in forge code is empty", "clean_source_pc")

        if(len(tampered_pc) !=0):
            publish_pc2(tampered_pc, "/decode_pc_tampered")
        else:
            rospy.logerr("%s in decode code is empty", "tampered_pc")

        if(len(tampered_points) !=0):
            publish_pc2(tampered_points, "/decode_tampered_points") 
        else:
            rospy.logerr("%s in forge code is empty", "tampered_points")
        
        if(len(generated_boundingbox) !=0):
            publish_pc2(generated_boundingbox, "/decode_generated_boundingbox")
        else:
            rospy.logerr("%s in encode is empty", "generated_boundingbox")

        if(len(gt_boundingbox) !=0):
            publish_pc2(gt_boundingbox, "/decode_gt_boundingbox")
        else:
            rospy.logerr("%s in encode is empty", "gt_boundingbox") 
        
        print('spin count', i)
        i += 1
    

    rospy.spin()