#!/usr/bin/env python
import numpy as np
import time
import math
from decimal import Decimal
from QIM_helper import *

#pc = np.array(  [[1.13,1.08, 1.02],
#                [1.03, 1.05, 1.04],
#                [1.09,1.01, 1.12],
#                [1.01, 1.07, 1.03]]).astype(float)


#global variables
                
pc_even = np.array(  [[1.12, 0.04, 1.32],
[1.42, 1.52, 1.62],
[1.72,1.82, 1.92],
[1.14,1.24, 1.34],
[1.44, 1.54, 1.64],
[1.74,1.84, 1.94],
[1.16,1.26, 1.36],
[1.46, 1.56, 1.66],
[1.76,1.86, 1.96],
[1.18,1.28, 1.38],
[1.48, 1.58, 1.68],
[1.78,1.88, 1.98],
[1.15,1.25, 1.35],
[1.45, 1.55, 1.65],
[1.75,1.85, 1.95],
[1.10,1.20, 1.30],
[1.40, 1.50, 1.60],
[1.70,1.80, 1.90],
[3.70,4.9324, 4.90],
[9.789,1.245, 5.674],
[0.070,0.080, 0.090]
]).astype(float)


pc_odd = np.array(  [[1.13, 1.23, 1.33],
[1.43, 1.53, 1.63],
[1.73,1.83, 1.93],
[1.11,1.21, 1.31],
[1.41, 1.51, 1.61],
[1.71,1.81, 1.91],
[1.17,1.27, 1.37],
[1.47, 1.57, 1.67],
[1.77,1.87, 1.97],
[1.19,1.29, 1.39],
[1.49, 1.59, 1.69],
[1.79,1.89, 1.99],
[1.15,1.25, 1.35],
[1.45, 1.55, 1.65],
[1.75,1.85, 1.95],
[1.10,1.20, 1.30],
[1.40, 1.50, 1.60],
[1.70,1.80, 1.90]]).astype(float)


pc_test = np.array([[52.249, 12.482,  2.026],
       [51.769, 12.453,  2.01 ],
       [51.275, 12.505,  1.994],
       [50.792, 12.556,  1.978],
       [50.31 , 12.604,  1.963],
       [49.84 , 12.653,  1.948],
       [49.363, 12.697,  1.933]]).astype(np.float64)

cb_1 = np.array([[0 ,1 ,0],
 [1 ,1 ,1],
 [1 ,1 ,0],
 [0 ,0 ,0],
 [0 ,1 ,1],
 [0 ,1 ,0],
 [0 ,1, 0],
 [0 ,0, 0],
 [1 ,0, 0],
 [1 ,1, 1]])

cb_2 = np.array([[0 ,1 ,0],
 [1 ,1 ,1],
 [1 ,1 ,0],
 [0 ,0 ,0],
 [0 ,1 ,1],
 [0 ,1 ,0],
 [1 ,1, 1],
 [1 ,1, 1],
 [0 ,1, 0],
 [0 ,0, 0],
 [1 ,0, 0],
 [1 ,1, 1],
 [1 ,1, 1]])

def getPointCloud_from_quantizedValues(pc_encoded, resolution_in):
    pc_value = (pc_encoded *resolution_in))
    return(pc_value)

def getQuantizedValues_from_pointCloud(pc, resolution_in):
    pc_quant_value = np.around((pc - np.array([x_in[0],y_in[0],z_in[0]])) / resolution_in).astype(np.int32)
    # pc_quant_value = ((pc - np.array([x_in[0],y_in[0],z_in[0]]))/resolution_in).astype(np.int32)
    return(pc_quant_value)

#function to quantize one row based on the input message bits




def Q0_quantization(pc, delta):
    pc_quant_value[0] = np.around(pc[0]/delta)*delta.astype(np.float)
    return(pc_quant_value)

def Q1_quantization(pc, delta):
    pc_quant_value = (np.around(pc-(delta/2)/delta) + 0.5)*delta.astype(np.float)
    return(pc_quant_value)
 

if __name__ == '__main__':


    # 1. encode the point cloud and extract the codebook

    pc_input = pc_test
    

    print('point cloud shape and values', pc_input.shape, pc_input)
    # c = np.around(((pc_input - np.array([x[0],y[0],z[0]])) / resolution_delta)).astype(np.int32)
    # print('quantized pc numpy representation', c)

    # quantized_pc  = c*resolution_delta + np.array([x[0],y[0],z[0]])
    # print('pc values', quantized_pc) 

    # voxel_delta, voxel_halfdelta, encoded_CB = qim_quantize_restricted_threebits(pc_input)

    voxel_delta, voxel_halfdelta, encoded_CB = qim_quantize_restricted_threebits_new(pc_input)
    
    
    voxel_halfdelta_npy = np.array([voxel_halfdelta]).reshape(-1,3)
    
    
    print('voxel_halfdelta_npy representation', voxel_halfdelta_npy)
    print('voxel delta', np.array(voxel_delta))
    # print('voxel_halfdelta', voxel_halfdelta)

    # 2. Get the PC from the quantized values 
    # #encoded_quantized_pc  = encoded_pc*resolution + np.array([x[0],y[0],z[0]])
    encoded_quantized_pc  = getPointCloud_from_quantizedValues(voxel_halfdelta_npy, resolution_halfdelta, x,y,z)
    
    print('qim pc values', encoded_quantized_pc) 
    # d = ((pc_input - np.array([x[0],y[0],z[0]])) / resolution).astype(np.int32)

    # # 3. Get the decoded codebook and quantized values of decoded point cloud   
    # # Decode the point cloud and get the code book
    
    # decoded_CB, decoded_quantized_values = qim_decode(encoded_quantized_pc, resolution_halfdelta)

    decoded_CB = qim_decode_new(encoded_quantized_pc, resolution_delta, x,y,z)

    # print('decoded pc numpy representation', np.array([decoded_quantized_values]).reshape(-1,3))

    decoded_codebook = np.array([decoded_CB]).reshape(-1,3)
    encoded_codebook = np.array([encoded_CB]).reshape(-1,3)

    print('decoded_codebook', decoded_codebook)
    print('encoded_codebook', encoded_codebook)

    # #compare codebooks
    compare_codebooks(encoded_codebook, decoded_codebook)

    # threshold =  0.7
    # block_size = 2

    # cb_1 = np.random.randint(2, size = [10, 3])
    #print('cb_1:', cb_1)
    # cb_2 = np.random.randint(2, size = [13, 3])
    #print('cb_2:', cb_2)


    # max_size = max(cb_1.shape[0],cb_2.shape[0] )

    # window_size = 2
    # print('max pc size', max_size)
    # iter_count = int(max_size/window_size)
    # print('iter count:', iter_count)

    # Get_tamperindices_blockbased(decoded_code)
    
    # suspect_index = Get_tamperindices_pointbased(decoded_code)
    # print('culprit indices', np.array([suspect_index]))

    # calculate_diff(cb_1,cb_2,block_size,threshold)



