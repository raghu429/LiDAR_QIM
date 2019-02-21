#!/usr/bin/env python
import sys
import numpy as np
import ntpath
import shutil

from helper_functions import*



if __name__ == '__main__':
        
        data_directory = './QIM_data/test_img/'
        
        source_directory = '../../KittiDataset/data_object_calib/training/calib/'
        destination_dir = './QIM_data/calib/'
        
        # source_directory = '../../KittiDataset/training/label_2/'
        # destination_dir = './QIM_data/label_2/'
        
        
        # source_directory = '../../KittiDataset/data_object_velodyne/training/velodyne/'
        # destination_dir = './QIM_data/test_data/'
        
        
        

        for filename in os.listdir(data_directory):
                # if filename.endswith(".bin")  and filename.startswith("000014"):
                if filename.endswith(".png") :
                        working_file_name = ntpath.basename(os.path.join(data_directory, filename)).split('.')[0]
                        print('file currently working on %s'%(working_file_name) )
                        
                        shutil.copy2(source_directory+working_file_name+'.txt', destination_dir)

                        # shutil.copy2(source_directory+working_file_name+'.bin', destination_dir)
