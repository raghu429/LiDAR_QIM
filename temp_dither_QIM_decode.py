#!/usr/bin/env python
import sys
import numpy as np
# np.set_printoptions(threshold=np.nan)
# import pcl
import rospy
import ntpath
import sensor_msgs.point_cloud2 as pc2
import shutil
import re
from scipy.stats import pearsonr
from sensor_msgs.msg import PointCloud2, PointField
from random import randint
#import settings
from helper_functions import *
from tamper_pc import *
from QIM_helper import *
from dither_randomscratchpad import *


forged_data_directory = './QIM_data/forged_Dither/'
encoded_data_directory = './QIM_data/encoded_Dither/'
clean_data_directory = './QIM_data/camfiltered/'
label_dir = './QIM_data/label_2/'
calib_dir = './QIM_data/calib/'
img_dir = './QIM_data/test_img/'
unwanted_img_dir = './QIM_data/test_img_unwanted/'

op_dir = './QIM_data/temp_decode_results_ditherexperiment/'
if not os.path.exists(op_dir):
  os.mkdir(op_dir)

#************************************************
#************************************************
# All globals are initialized in QIM_helper.py


def move_gt_label(cluster_to_copy,yrange, ymax):
    #get the cluster to be replicated
    displacement = yrange*2
   
    if (ymax < 0):
        #modify the cluster
        cluster_to_copy[:,1] = cluster_to_copy[:,1] + displacement
    else:
        #modify the cluster
        cluster_to_copy[:,1] = cluster_to_copy[:,1] - displacement
        
    return cluster_to_copy

if __name__ == '__main__':

    # ROS node initialization
    rospy.init_node('watermark_decode', anonymous = True)
 
    filecount = 0
    match_string = re.compile(r'sigma_0-0072_uniform_add_bs[0-9]*_dr3_[0-9]*.npy')

    for filename in os.listdir(forged_data_directory):
        
        tampered_pc = []
        clean_source_pc = []
        tampered_points = []
        generated_boundingbox = []
        gt_boundingbox = []

        #results
        glb_FN_count = 0
        glb_FP_count = 0
        corr_pear = 0
        volume_difference = 0
        distortion = 0
        b_errorRate = 0
 

        # if filename.endswith(".npy") and filename.startswith("001179"):
        # if filename.endswith(".npy"):
        
        # if(filename == 'sigma_0-0072_uniform_add_bs128_dr3_000020.npy'):
        if(match_string.match(filename)):
            print('file name', filename) 
            # break  
            split_file_name = []
            working_file_name = []
            
            working_file_name = ntpath.basename(os.path.join(forged_data_directory, filename)) 
            
            ## splits the file name into an array of elements
            ## for example : sigma_0-05_uniform_del_bs256_dr2_000125.npy 
            ## is split into seven string elements so split_file_name[6] will be the 000125.npy
            ## ['sigma', '0-05', 'uniform', 'del', 'bs256', 'dr2', '000125.npy']
            split_file_name = working_file_name.split('_')

            
            # print('file currently working on %s'%(working_file_name))
            # print('split file name', split_file_name )
             
            #get the cluster source point cloud
            global_file_name = split_file_name[6]
            # print('global file name', global_file_name) 
            clean_source_pc = np.load(os.path.join(clean_data_directory, global_file_name))
            encoded_filename  = split_file_name[4] + '_' + split_file_name[5] + '_' + split_file_name[6]

            print('encoded file name', encoded_filename)
            # read the input tampered point cloud
            print('loading file: from ', working_file_name, os.path.join(forged_data_directory, working_file_name))
            tampered_pc = np.load(os.path.join(forged_data_directory, working_file_name))
            print('loaded file size', tampered_pc.size)
            print('loaded file shape', tampered_pc.shape)

            block_size = float(re.findall('\d+', split_file_name[4])[0])
            range_factor = float(re.findall('\d+', split_file_name[5])[0])
            print('range factor, block_size', range_factor, block_size)
            decode_ = qim_decode_dither(np.copy(tampered_pc), resolution_delta, block_size,range_factor)

            decoded_cb = np.asarray(decode_).reshape(-1,1)
            length_pc = len(tampered_pc)
            print('length of pc', length_pc)
            encode_ = encode_bitstream(block_size, length_pc)

            encoded_cb = np.asarray(encode_).reshape(-1,1)

        #         #****************************************************************************************************************************
        #                         # BER
        #                  #****************************************************************************************************************************

            # tampered_indices, b_errorRate = get_tamperedindices_dither_threebits(decoded_cb.reshape(-1,3), block_size, length_pc, 'full')
            
            ## half would only count the row as error if the number of different bits is greater than 1. For ex: encode: 101 & decode: 111 then half wont count it as an error index where as the full would.

            tampered_indices, b_errorRate = get_tamperedindices_dither_threebits(decoded_cb.reshape(-1,3), block_size, length_pc, 'half')
            
            # BER_final.append(b_errorRate)
            # tampered_indices = np.array([tampered_indices])
            
            print('length of suspect indices', tampered_indices.shape[0], tampered_indices)
            # print('suspect indice length', tampered_indices.shape[0], tampered_indices)
        
        #         #****************************************************************************************************************************
        #                         # Correlation
        #         #****************************************************************************************************************************


            corr_pear = pearsonr(encoded_cb, decoded_cb)[0]
            
            # correlation_final.append(corr_pear)

            

        #         #****************************************************************************************************************************
        #                         # Localization Accuracy
        #         #****************************************************************************************************************************

            label_file_name = global_file_name.split('.')[0] + '.txt'
            # print('calib file name', label_file_name)

            if(len(tampered_indices) != 0): #check to see we got an intact point cloud
                
                if('clean' in working_file_name):
                    glb_FP_count = 1
                else:
                    glb_FP_count = 0
                    if(len(tampered_indices) < 2):
                        distortion = 75

                if('del' in working_file_name):
                    # read the reference input cloud. Since the tampered pc wont have any points at these locations, if we extract points from it, we'll get no corners.
                    reference_pc = np.load(os.path.join(encoded_data_directory, encoded_filename))
                    tampered_points = reference_pc[tampered_indices]
                else:
                    tampered_points = tampered_pc[tampered_indices]

                
                tampered_points =  np.array(tampered_points)
                print("tampered points", tampered_points.shape, tampered_points)
                print("tampered points", tampered_points.shape)
                
        #         #GT bounding box 
                label_file_name = global_file_name.split('.')[0] + '.txt'
                calib_file_name = global_file_name.split('.')[0] + '.txt'

                print('calib file name', calib_file_name)
                #read label and get the gt_bounding box corners
                places, rotation, sizes = read_labels(os.path.join(label_dir, label_file_name), 'txt', calib_path = None , is_velo_cam=False, proj_velo=None)
            
                if(places is None):
                    print('no labels found')
                    # shutil.move(img_dir+working_file_name+'.png', unwanted_img_dir)
        #             continue

                gt_corners = visualize_groundtruth(os.path.join(calib_dir,calib_file_name), os.path.join(label_dir, label_file_name))
                print('corners', gt_corners.shape, gt_corners[0])

                if(gt_corners.shape[0]):
                    #pick a label .. here we are  picking label 0 .. just to be on the same side
                    logical_bound, x_range, y_range, z_range, y_max = get_cluster_logicalbound (np.copy(clean_source_pc), gt_corners[label_tomove_index])
                    a = np.array([np.where(logical_bound == True)])
                    print('logical bound shape', a.shape)
                else:
                    print("GT label list is empty")
                    continue
                    
                #generate bounding box based on shifting the co-ordinates from original label (this is like ground truth)
                if('add' in working_file_name ):
                    gt_boundingbox = move_gt_label(np.copy(gt_corners[label_tomove_index]), y_range, y_max)
                else:
                    gt_boundingbox = gt_corners[label_tomove_index]


                gt_boundingbox = gt_boundingbox.reshape(-1,3)
                print('Groundtruth bb', gt_boundingbox)
                # gt_boundingbox = np.array([bb_a[0], bb_a[1], bb_a[2], bb_a[3]])
            
                #Generated bounding box
                generated_boundingbox = get_boundingboxcorners(tampered_points)
                print('generated bb', generated_boundingbox)
                generated_boundingbox = generated_boundingbox.reshape(-1,3)
                # print("bb_b shape", bb_b.shape)
                # generated_boundingbox = np.array([bb_b[0], bb_b[1], bb_b[2], bb_b[5]])

                #calculate IoU
                # overlap_area = bb_intersection_over_union(generated_boundingbox[:,:2], gt_boundingbox[:,:2])
                # print("IOU", overlap_area)
                volume_difference = abs(get_volume_fromcorners(generated_boundingbox)- get_volume_fromcorners(gt_boundingbox))

                distortion = Hausdorff_dist(generated_boundingbox, gt_boundingbox )
                

            else:
                print('no tampered indices not found')
                if('clean' not in working_file_name):
                    glb_FN_count = 1
                    distortion = 100
                else:
                    glb_FN_count = 0
            
                    # shutil.move(img_dir+working_file_name+'.png', unwanted_img_dir)
        
            #      #****************************************************************************************************************************
            #                         # Output
            #         #****************************************************************************************************************************
            #the output file name will have the same front portion as an input file. Lie for ex: an input file sigma_0-05_uniform_del_bs256_dr2_000125.npy will have the following output files for BER: sigma_0-05_uniform_del_bs256_dr2_000125_BER.npy. Similar trend for distortion, correlation and False Alarm 
            
            file_to_save = working_file_name.split('.')[0] 
            
            op_filename_BER = file_to_save+'_BER.npy'
            op_filename_DIST = file_to_save+'_DIST.npy'
            op_filename_CORR = file_to_save+'_CORR.npy'
            # op_filename_VOL = file_to_save+'_VOL.npy'
            op_filename_FA = file_to_save+'_FA.npy'
            op_filename_FN = file_to_save+'_FN.npy'

            np.save(os.path.join(op_dir, op_filename_BER), b_errorRate)
            np.save(os.path.join(op_dir, op_filename_DIST), distortion)
            np.save(os.path.join(op_dir, op_filename_CORR), corr_pear)
            np.save(os.path.join(op_dir, op_filename_FN), glb_FN_count)
            np.save(os.path.join(op_dir, op_filename_FA), glb_FP_count)

            # console output
            # print('********************************************************')
            # print('\n')
            # print('file name', op_filename)
            # print('File count:', glb_file_count)
            print ('FN:', glb_FN_count)
            print ('FP:', glb_FP_count)
            # print('temp array', temp_array)
            print('correlation', corr_pear)
            print('BER', b_errorRate)
            # print('Vol', volume_final)
            print('distortion', distortion)
            filecount += 1
            print(filecount)
            
            # break
            
        
        # else:

            # print('no file found')
    
        
        

    i = 0
    print ('***********************************')
    
    while not rospy.is_shutdown():

        # if(len(clean_source_pc) !=0):
        #     publish_pc2(clean_source_pc, "/qim_tamper_clean_pc")
        # else:
        #     rospy.logwarn("%s in forge code is empty", "clean_source_pc")

        # if(len(tampered_pc) !=0):
        #     publish_pc2(tampered_pc, "/decode_pc_tampered")
        # else:
        #     rospy.logerr("%s in decode code is empty", "tampered_pc")

        # if(len(tampered_points) !=0):
        #     publish_pc2(tampered_points, "/decode_tampered_points") 
        # else:
        #     rospy.logerr("%s in forge code is empty", "tampered_points")
        
        # if(len(generated_boundingbox) !=0):
        #     publish_pc2(generated_boundingbox, "/decode_generated_boundingbox")
        # else:
        #     rospy.logerr("%s in encode is empty", "generated_boundingbox")

        # if(len(gt_boundingbox) !=0):
        #     publish_pc2(gt_boundingbox, "/decode_gt_boundingbox")
        # else:
        #     rospy.logerr("%s in encode is empty", "gt_boundingbox") 
        
        print('spin count', i)
        i += 1
        break
    

    rospy.spin()