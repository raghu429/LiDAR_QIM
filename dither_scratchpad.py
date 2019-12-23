# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#!/usr/bin/env python
import numpy as np
import time
import math
from decimal import Decimal

#


pc_test = np.array([[52.249, 12.482,  2.026],
       [51.769, 12.453,  2.01 ],
       [51.275, 12.505,  1.994],
       [50.792, 12.556,  1.978],
       [50.31 , 12.604,  1.963],
       [49.84 , 12.653,  1.948],
       [49.363, 12.697,  1.933],
       [60.72, 18.35,  1.54]]).astype(np.float64)

def qim_encode_dither(pc_input, resolution_delta):    
    #for [0 0 0 0] modify nothing
    c_out = np.empty((0,3), np.float64)
    c = np.array([0.0,0.0,0.0]).astype(np.float64)  
    count = 0

    message = 0
    for i in np.ndindex(pc_input.shape[:-1]):
        print('i', i,pc_input[i])
        if(message == 0):
            c[0] = quantization_nodelta(pc_input[i][0], resolution_delta)            
            c[1] = quantization_nodelta(pc_input[i][1], resolution_delta)          
            c[2] = quantization_nodelta(pc_input[i][2], resolution_delta)
            
#            print('message', message)
#            print('c[0]', c[0])                
#            print('c[1]', c[1])            
#            print('c[2]', c[2])
        elif(message == 1): #[001] modify z
            c[0] = quantization_nodelta(pc_input[i][0], resolution_delta)        
            c[1] = quantization_nodelta(pc_input[i][1], resolution_delta)
            c[2] = quantization_add_delta(pc_input[i][2], resolution_delta) 
            #c[2] = (np.around(((pc_input[i][2]) / resolution_delta))* resolution_delta + (resolution_delta/2.0)).astype(np.float64)
#            print('message', message)            
#            print('c[0]', c[0])                
#            print('c[1]', c[1])            
#            print('c[2]', c[2])    
        elif(message == 2): #[010] modify y
            c[0] = quantization_nodelta(pc_input[i][0], resolution_delta)        
            c[1] = quantization_add_delta(pc_input[i][1], resolution_delta)
            c[2] = quantization_nodelta(pc_input[i][2], resolution_delta)
#            print('message', message)            
#            print('c[0]', c[0])                
#            print('c[1]', c[1])            
#            print('c[2]', c[2])
        elif(message == 3): #[011] modify y,z
            c[0] = quantization_nodelta(pc_input[i][0], resolution_delta)        
            c[1] = quantization_add_delta(pc_input[i][1], resolution_delta)
            c[2] = quantization_add_delta(pc_input[i][2], resolution_delta)
#            print('message', message)            
#            print('c[0]', c[0])                
#            print('c[1]', c[1])            
#            print('c[2]', c[2])    
        elif(message == 4): #[100] modify x
            c[0] = quantization_add_delta(pc_input[i][0], resolution_delta)
            c[1] = quantization_nodelta(pc_input[i][1], resolution_delta)   
            c[2] = quantization_nodelta(pc_input[i][2], resolution_delta)
#            print('message', message)            
#            print('c[0]', c[0])                
#            print('c[1]', c[1])            
#            print('c[2]', c[2])
        elif(message == 5): #[101] modify y,z
            c[0] = quantization_add_delta(pc_input[i][0], resolution_delta)
            c[1] = quantization_nodelta(pc_input[i][1], resolution_delta) 
            c[2] = quantization_add_delta(pc_input[i][2], resolution_delta)
#            print('message', message)            
#            print('c[0]', c[0])                
#            print('c[1]', c[1])            
#            print('c[2]', c[2])
        elif(message == 6): #[110] modify x,y
            c[0] = quantization_add_delta(pc_input[i][0], resolution_delta)
            c[1] = quantization_add_delta(pc_input[i][1], resolution_delta)
            c[2] = quantization_nodelta(pc_input[i][2], resolution_delta)
#            print('message', message)            
#            print('c[0]', c[0])                
#            print('c[1]', c[1])            
#            print('c[2]', c[2])
        elif(message == 7): #[011] modify x,y,z
            c[0] = quantization_add_delta(pc_input[i][0], resolution_delta)
            c[1] = quantization_add_delta(pc_input[i][1], resolution_delta)
            c[2] = quantization_add_delta(pc_input[i][2], resolution_delta)
#            print('message', message)            
#            print('c[0]', c[0])                
#            print('c[1]', c[1])            
#            print('c[2]', c[2])
        else:
             print('check the input message')
        
        count+=1
        #print('count,message', count, message)
        message += 1
        message = message % 8
        c_out = np.append(c_out, [c], axis = 0)
        #print('c_out', c_out)
        
    return(np.array(c_out).astype(np.float64))


def qim_encode_old(pc_input, resolution_delta):    
    #for [0 0 0 0] modify nothing
    c_out = np.empty((0,3), np.float64)
    c = np.array([0.0,0.0,0.0]).astype(np.float64)  
    count = 0

    message = 0
    for i in np.ndindex(pc_input.shape[:-1]):
        print('i', i,pc_input[i])
        if(message == 0):
            c[0] = 2*quant_(pc_input[i][0], resolution_delta)            
            c[1] = 2*quant_(pc_input[i][1], resolution_delta)          
            c[2] = 2*quant_(pc_input[i][2], resolution_delta)
            
#            print('message', message)
#            print('c[0]', c[0])                
#            print('c[1]', c[1])            
#            print('c[2]', c[2])
        elif(message == 1): #[001] modify z
            c[0] = 2*quant_(pc_input[i][0], resolution_delta)        
            c[1] = 2*quant_(pc_input[i][1], resolution_delta) 
            c[2] = 2*quant_(pc_input[i][2], resolution_delta) + 1 
            #c[2] = (np.around(((pc_input[i][2]) / resolution_delta))* resolution_delta + (resolution_delta/2.0)).astype(np.float64)
#            print('message', message)            
#            print('c[0]', c[0])                
#            print('c[1]', c[1])            
#            print('c[2]', c[2])    
        elif(message == 2): #[010] modify y
            c[0] = 2*quant_(pc_input[i][0], resolution_delta)        
            c[1] = 2*quant_(pc_input[i][1], resolution_delta) + 1
            c[2] = 2*quant_(pc_input[i][2], resolution_delta)
#            print('message', message)            
#            print('c[0]', c[0])                
#            print('c[1]', c[1])            
#            print('c[2]', c[2])
        elif(message == 3): #[011] modify y,z
            c[0] = 2*quant_(pc_input[i][0], resolution_delta)        
            c[1] = 2*quant_(pc_input[i][1], resolution_delta) + 1 
            c[2] = 2*quant_(pc_input[i][2], resolution_delta) + 1
#            print('message', message)            
#            print('c[0]', c[0])                
#            print('c[1]', c[1])            
#            print('c[2]', c[2])    
        elif(message == 4): #[100] modify x
            c[0] = 2*quant_(pc_input[i][0], resolution_delta) + 1 
            c[1] = 2*quant_(pc_input[i][1], resolution_delta)   
            c[2] = 2*quant_(pc_input[i][2], resolution_delta)
#            print('message', message)            
#            print('c[0]', c[0])                
#            print('c[1]', c[1])            
#            print('c[2]', c[2])
        elif(message == 5): #[101] modify y,z
            c[0] = 2*quant_(pc_input[i][0], resolution_delta) +1 
            c[1] = 2*quant_(pc_input[i][1], resolution_delta) 
            c[2] = 2*quant_(pc_input[i][2], resolution_delta) +1
#            print('message', message)            
#            print('c[0]', c[0])                
#            print('c[1]', c[1])            
#            print('c[2]', c[2])
        elif(message == 6): #[110] modify x,y
            c[0] = 2*quant_(pc_input[i][0], resolution_delta) +1
            c[1] = 2*quant_(pc_input[i][1], resolution_delta)+1
            c[2] = 2*quant_(pc_input[i][2], resolution_delta)
#            print('message', message)            
#            print('c[0]', c[0])                
#            print('c[1]', c[1])            
#            print('c[2]', c[2])
        elif(message == 7): #[111] modify x,y,z
            c[0] = 2*quant_(pc_input[i][0], resolution_delta) +1 
            c[1] = 2*quant_(pc_input[i][1], resolution_delta)+1
            c[2] = 2*quant_(pc_input[i][2], resolution_delta)+1
#            print('message', message)            
#            print('c[0]', c[0])                
#            print('c[1]', c[1])            
#            print('c[2]', c[2])
        else:
             print('check the input message')
        
        count+=1
        #print('count,message', count, message)
        message += 1
        message = message % 8
        c_out = np.append(c_out, [c], axis = 0)
        #print('c_out', c_out)
        
    return(np.array(c_out*(resolution_delta/2.0)).astype(np.float64))



def quantization_add_delta(val, step):
    quantized_value = (np.around(val/step)*step + (step/2.0)).astype(np.float64)
    return(quantized_value)    

def quantization_add_delta_decode(val, step):
    quantized_value = (np.around((val-(step/2.0))/step)*step + (step/2.0)).astype(np.float64)
    return(quantized_value)

def quantization_nodelta(val, step):
    quantized_value = (np.around(val/step)*step).astype(np.float64)
    return(quantized_value)

def quantization_halfdelta(val, step_full):
    step = step_full/2.0
    quantized_value = (np.around(val/step)*step).astype(np.float64)
    return(quantized_value)

def quant_(val, step):
    quantized_value = (np.around(val/step)).astype(np.int64)
    return(quantized_value)


    
def qim_decode_dither(pc_input,resolution_delta):

        
        message_decoded = []
        #print('c_dist shape', c_dist)
        # print('message', message)            
        
        for i in np.ndindex(pc_input.shape[:-1]):
            c_dist = []
            
            print('*********************')    
            print('i',i)
            c = np.array([0.0,0.0,0.0]).astype(np.float64)
            c[0] = quantization_nodelta(pc_input[i][0], resolution_delta)            
            c[1] = quantization_nodelta(pc_input[i][1], resolution_delta)
            c[2] = quantization_nodelta(pc_input[i][2], resolution_delta)
            print ('C', c)

            d = np.array([0.0,0.0,0.0]).astype(np.float64)  
            d[0] = quantization_halfdelta(pc_input[i][0], resolution_delta)
            d[1] = quantization_halfdelta(pc_input[i][1], resolution_delta)
            d[2] = quantization_halfdelta(pc_input[i][2], resolution_delta) 
            print ('D', d)

            f = np.array([0.0,0.0,0.0]).astype(np.float64)  
            f[0] = quantization_add_delta_decode(pc_input[i][0], resolution_delta)
            f[1] = quantization_add_delta_decode(pc_input[i][1], resolution_delta)
            f[2] = quantization_add_delta_decode(pc_input[i][2], resolution_delta)

            print('F', f)

            print('-------------')    

            message = []
            if(abs(c[0]-d[0]) < abs(c[0]-f[0])):
                message.append(0)
            else:
                message.append(1)
            if(abs(c[1]-d[1]) < abs(c[1]-f[1])):
                message.append(0)
            else:
                message.append(1)
            if(abs(c[2]-d[2]) < abs(c[2]-f[2])):
                message.append(0)
            else:
                message.append(1)
            
            message_decoded.append(message)

        print('message decoded', message_decoded)

            
        
        
    #     #Q0 

    #         c = np.array([0.0,0.0,0.0]).astype(np.float64) 
    #         # c[0] = pc_input[i][0]
    #         # c[1] = pc_input[i][1]
    #         # c[2] = pc_input[i][2]

    #         c[0] = quantization_nodelta(pc_input[i][0], resolution_delta)            
    #         c[1] = quantization_nodelta(pc_input[i][1], resolution_delta)
    #         c[2] = quantization_nodelta(pc_input[i][2], resolution_delta)
    # #        print('message', message)
    # #        print('c[0]', c[0])                
    # #        print('c[1]', c[1])            
    # #        print('c[2]', c[2])
    #         print('i', i)
    #         c_dist.append(distance_manhattan(c, 'xyz', pc_input[i]))             
    #     #Q1
    #         c = np.array([0.0,0.0,0.0]).astype(np.float64) 
    #         c[0] = quantization_nodelta(pc_input[i][0], resolution_delta)        
    #         c[1] = quantization_nodelta(pc_input[i][1], resolution_delta)
    #         # c[0] = pc_input[i][0]
    #         # c[1] = pc_input[i][1]
    #         c[2] = quantization_halfdelta(pc_input[i][2], resolution_delta) 
    # #        print('message', message)            
    # #        print('c[0]', c[0])                
    # #        print('c[1]', c[1])            
    # #        print('c[2]', c[2])    
    #         #print('i', i)            
    #         c_dist.append(distance_manhattan(c, 'z', pc_input[i]))
    #     #Q2
    #         c = np.array([0.0,0.0,0.0]).astype(np.float64)  
    #         c[0] = quantization_nodelta(pc_input[i][0], resolution_delta)
    #         # c[0] = pc_input[i][0]
    #         c[1] = quantization_halfdelta(pc_input[i][1], resolution_delta)
    #         # c[2] = pc_input[i][2]
    #         c[2] = quantization_nodelta(pc_input[i][2], resolution_delta)
    # #        print('message', message)            
    # #        print('c[0]', c[0])                
    # #        print('c[1]', c[1])            
    # #        print('c[2]', c[2])
    #         #print('i', i)            
    #         c_dist.append(distance_manhattan(c,'y',pc_input[i]))
    #     #Q3
    #         c = np.array([0.0,0.0,0.0]).astype(np.float64)         
    #         c[0] = quantization_nodelta(pc_input[i][0], resolution_delta)        
    #         # c[0] = pc_input[i][0]
    #         c[1] = quantization_halfdelta(pc_input[i][1], resolution_delta)
    #         c[2] = quantization_halfdelta(pc_input[i][2], resolution_delta)
    # #        print('message', message)            
    # #        print('c[0]', c[0])                
    # #        print('c[1]', c[1])            
    # #        print('c[2]', c[2])
    #         c_dist.append(distance_manhattan(c,'yz', pc_input[i]))
    #     #Q4        
    #         c = np.array([0.0,0.0,0.0]).astype(np.float64)  
    #         c[0] = quantization_halfdelta(pc_input[i][0], resolution_delta)
    #         # c[1] = pc_input[i][1]
    #         # c[2] = pc_input[i][2]
    #         c[1] = quantization_nodelta(pc_input[i][1], resolution_delta)
    #         c[2] = quantization_nodelta(pc_input[i][2], resolution_delta)
    # #        print('message', message)            
    # #        print('c[0]', c[0])                
    # #        print('c[1]', c[1])            
    # #        print('c[2]', c[2])
    #         c_dist.append(distance_manhattan(c,'x',pc_input[i]))
    #     #Q5        
    #         c = np.array([0.0,0.0,0.0]).astype(np.float64) 
    #         c[0] = quantization_halfdelta(pc_input[i][0], resolution_delta)
    #         c[1] = quantization_nodelta(pc_input[i][1], resolution_delta) 
    #         # c[1] = pc_input[i][1]
    #         c[2] = quantization_halfdelta(pc_input[i][2], resolution_delta)
    # #        print('message', message)            
    # #        print('c[0]', c[0])                
    # #        print('c[1]', c[1])            
    # #        print('c[2]', c[2])
    #         c_dist.append(distance_manhattan(c, 'xz',pc_input[i]))
    #     #Q6 
    #         c = np.array([0.0,0.0,0.0]).astype(np.float64)  
    #         c[0] = quantization_halfdelta(pc_input[i][0], resolution_delta)
    #         c[1] = quantization_halfdelta(pc_input[i][1], resolution_delta)
    #         c[2] = quantization_nodelta(pc_input[i][2], resolution_delta)
    #         # c[2] = pc_input[i][2]       
    # # print('message', message)            
    # #        print('c[0]', c[0])                
    # #        print('c[1]', c[1])            
    # #        print('c[2]', c[2])
    #         c_dist.append(distance_manhattan(c, 'xy', pc_input[i]))
    #     #Q7
    #         c = np.array([0.0,0.0,0.0]).astype(np.float64)  
    #         c[0] = quantization_halfdelta(pc_input[i][0], resolution_delta)
    #         c[1] = quantization_halfdelta(pc_input[i][1], resolution_delta)
    #         c[2] = quantization_halfdelta(pc_input[i][2], resolution_delta) 
    # #       print('message', message)            
    # #        print('c[0]', c[0])                
    # #        print('c[1]', c[1])            
    # #        print('c[2]', c[2])
    #         c_dist.append(distance_manhattan(c, 'xyz', pc_input[i]))
            
    #         #find the min distance
    #         a_min_index = np.where(c_dist == np.min(c_dist))
    #         message_decoded.append(a_min_index[0][0])
    #     return(message_decoded)
    
# def distance_manhattan(pc1, mod_type ,pc2):
#     print('-------------')    
#     print('pc1', pc1)    
#     print('pc2', pc2)

#     print('first', abs(pc1[0].astype(np.float64) - pc2[0].astype(np.float64)))
#     print('second', abs(pc1[1].astype(np.float64) - pc2[1].astype(np.float64)))
#     print('third', abs(pc1[2].astype(np.float64) - pc2[2].astype(np.float64)))

#     if(mod_type == 'xyz'):
#         # print('xzy')
#         dist = (abs(pc1[0].astype(np.float64) - pc2[0].astype(np.float64)) + abs(pc1[1].astype(np.float64) - pc2[1].astype(np.float64)) + abs(pc1[2].astype(np.float64) - pc2[2].astype(np.float64))).astype(np.float64)
#     elif(mod_type == 'x'):
#         # print('x')        
#         dist = (abs(pc1[0].astype(np.float64) - pc2[0].astype(np.float64))).astype(np.float64)
#     elif(mod_type == 'y'):
#         # print('y')
#         dist = (abs(pc1[1].astype(np.float64) - pc2[1].astype(np.float64))).astype(np.float64)
#     elif(mod_type == 'z'):
#         # print('z')
#         dist = (abs(pc1[2].astype(np.float64) - pc2[2].astype(np.float64))).astype(np.float64)
#     elif(mod_type == 'xy'):
#         # print('xy')
#         dist = (abs(pc1[0].astype(np.float64) - pc2[0].astype(np.float64)) + abs(pc1[1].astype(np.float64) - pc2[1].astype(np.float64))).astype(np.float64)
#     elif(mod_type == 'yz'):
#         # print('zy')
#         dist = (abs(pc1[2].astype(np.float64) - pc2[2].astype(np.float64)) + abs(pc1[1].astype(np.float64) - pc2[1].astype(np.float64))).astype(np.float64)
#     elif(mod_type == 'xz'):
#         # print('xz')
#         dist = (abs(pc1[0].astype(np.float64) - pc2[0].astype(np.float64)) + abs(pc1[2].astype(np.float64) - pc2[2].astype(np.float64))).astype(np.float64)
#     else:
#         print("bokka")
    
#     print('dist', dist)
#     print('****************')
    
#     print(dist)    
#     return(dist)


if __name__ == '__main__':


    # 1. encode the point cloud and extract the codebook

    pc_input = pc_test
    print('point cloud shape and values', pc_input.shape, pc_input)
    
    delta = 1
    pc_output_dither = qim_encode_dither(pc_input, delta)
    print('dither: output point cloud shape and values', pc_output_dither.shape, pc_output_dither)
    
    pc_output_old = qim_encode_old(pc_input, delta)
    print('old: output point cloud shape and values', pc_output_old.shape, pc_output_old)
    
    decode = qim_decode_dither(pc_output_dither, delta)
    print('decoded message', decode)