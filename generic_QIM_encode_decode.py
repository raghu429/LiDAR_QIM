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
from random import randint
from random import seed
from scipy.stats import pearsonr
from QIM_helper import qim_dummy_encoded_pc, get_tamperedindices_sequential_threebits,get_tamperedindices_sequential_twobits, get_tamperedindices_sequential_onebit
#


pc_test = np.array([ 40.30910863,   8.56577551,  54.46320516,  60.39623699,
         7.25475803,   4.65158945,  24.0168705 ,  68.55214697,
        69.95924573,  15.59284008,  17.31696879,  22.58139698,
        37.97602155,  23.98546057,  15.84086424,  64.47562428,
        64.68739303,  25.09177234,  68.35058661,  20.31693454,
        31.41407668,   7.74210555,   9.88657606,  13.63142911,
        49.15850831,  30.58308429,  28.70665896,  31.94941439,
         8.84467812,  55.1633896 ,  58.55935082,   0.37767247,
        19.88306825,  39.77105362,  66.51084712,  66.87295357,
        62.33949719,  47.45150092,  26.96712105,  46.75531802,
        29.20449607,  17.91013346,  13.92973487,  42.06037351,
         2.83635417,  58.57577831,  24.29099014,  21.64608291,
        38.06935796,  37.76095876,  60.17714093,  67.97861171,
        64.5394034 ,  14.40865039,  42.46893421,   9.8684628 ,
        32.20983921,   7.14781606,   9.81968821,   7.74585164,
        42.2642034 ,  39.67295414,  33.71234525,   5.65818973,
        64.67198355,   0.11937829,  52.89525163,  13.54698677,
        48.06953991,  39.07171809,  41.40008904,  48.51505537,
        32.02886082,  30.39700507,  46.70752464,  34.76806784,
        31.2269244 ,  41.47089453,   0.85778553,  44.13613895,
        43.13503661,  46.43440349,  47.12504245,  29.62829259,
        32.41893949,   7.47204453,  67.67971921,  49.86236027,
        41.963319  ,  34.12416954,  63.10778699,  63.89890794,
        15.49575045,  12.94987143,  62.06413481,  65.57312303,
        48.38139074,  26.62684837,  46.58874116,  69.61921899,
        63.89743138,  10.92567986,  56.93208869,  62.44335683,
        35.06506845,  68.32443014,  41.67924344,  13.36353113,
        34.33414038,  23.70018663,  33.56611469,  37.77336882,
        66.24683282,  54.81100422,  39.27624226,  12.03594921,
        14.77208202,  39.34060885,  43.97245292,  16.91737447,
        52.27097747,  63.27247195,   7.4567154 ,  30.42538059,
         5.67864239,  31.59656299,   4.83738733,  29.87560077,
        55.15963179,  54.76316068,  66.60179193,  40.85658816,
        15.2719692 ,  52.97308566,  54.8711188 ,  32.71282434,
        39.64884934,   7.33066156,  19.55792863,  64.60204768,
        31.426122  ,  51.22723749,  11.62464807,  20.00079172,
        39.69019735,  37.47606844,  56.2294566 ,  34.99068987,
        65.42478433,  56.22206364,   9.3182722 ,  61.51006047,
        15.90047396,  32.11596611,  15.77329698,  34.34906891,
        12.2422282 ,  10.16535032,  66.13204351,  51.5072004 ,
        30.2842393 ,  17.85495832,  48.98451946,  64.41068359,
        36.34781249,  42.74888539,  61.54246983,  66.30245282,
        36.05312667,  60.34839309,  54.81071621,  17.92651076,
        42.2639048 ,  57.3297187 ,  60.90814653,   1.28832283,
        19.61319887,   1.12418993,   2.14413319,  33.73727387,
         6.39290255,  50.25653794,  15.22536139,   8.16518441,
        43.13744103,  11.3460699 ,  10.91256245,  47.6356579 ,
         8.6844718 ,  20.22451467,  30.37347596,  53.25636839,
        46.91718731,  43.86829552,  48.28270364,  14.63490595,
        52.6422268 ,  49.48825693,  13.94900732,  34.9545124 ,
        15.26546121,  16.33555099,  37.39528069,   7.80218895,
        30.26773344,   4.27814462,   5.7914157 ,   6.73778133,
        19.39986213,  44.07643383,  31.49708685,   1.95598241,
        17.32229931,  57.91127847,   1.47590365,  18.13554504,
        48.20769933,  31.42970276,  68.28621842,  52.22813605,
        23.13992289,  50.85657027,  14.99210941,  55.26321535,
        14.1797156 ,   5.45952496,  48.81270969,  63.01683387,
        25.16365542,  47.98931001,  19.37827414,  33.07197612,
        48.49679153,  27.16663159,  28.91515434,  69.92270209,
        57.19264096,  49.12477051,  44.82754443,  61.90155248,
        24.72842686,  45.39266569,  33.17051334,   4.44022563,
        66.39424189,  45.01630971,   1.59195637,  36.66054374,
         1.54035206,  52.18984258,  12.65960367,  48.15929146,
        41.07184402,  12.07298663,  53.19537698,  10.70635002,
        25.05836709,  32.14612773,  41.87323068,  67.32080063,
        48.66122266,  20.05242194,  12.20315989,  22.70932629,
        60.90234432,   0.14143498,  56.98046841,   8.61064014,
        39.11400974,  66.95683921,  16.45231842,  47.95652755,
        64.60064791,  57.0939725 ,  52.59817392,  12.55326034,
        20.17932996,  61.50305811,  36.49333568,  24.58049503,
        58.63092406,  40.81717515,  23.18821241,  28.58986938,
        66.05816925,  65.94128619,  63.15945601,  53.92936809,
         3.45431239,   4.87568783,  32.82733725,  30.64270515,
        24.56345283,  53.36305542,  68.16478275,  49.77128387,
         8.7782343 ,  47.81074775,  24.2892622 ,  18.19833677]).astype(np.float64)


#task to make these functions generic and useful for 1D, 2D and 3D implementations and also for dither and normal QIM

#for encode_type = dither, the rate, rangefactor vary
#for encode_type = plain, the rate =1, rangefactor = 1
#for dimension = 1, 2 or 3 the message bits encoded in the PC vary
# usage
# for a 2 bit message '10'normal qim encoding: qim_encode_generic(pc_input, resolution_delta, rate, range_factor, 'normal', 2, 3):
#
#
#
#


def qim_encode_generic(pc_input, resolution_delta, rate, range_factor, encode_type, dimension, message_value):    
    #for [0 0 0 0] modify nothing
     
    

    # message = [7, 7, 7, 3, 3, 4, 0, 0 ]
    # message = [7, 7, 7, 7,7,7,7,7]

    #This message_value can be given a zero if you want the code to use the normal (0-7) for three bits, 0-3 for two bits and 0-1 for one bit. Or this can be given as an independant value to be embedded
    msg_val = message_value
    count = 0

    if(encode_type == 'normal'):
        rate = 1
        range_factor = 1

    if(dimension == 3):
        c_out = np.empty((0,3), np.float64)
        c = np.array([0.0,0.0,0.0]).astype(np.float64)
        message_range = 8
    elif(dimension == 2):
        c_out = np.empty((0,2), np.float64)
        c = np.array([0.0,0.0]).astype(np.float64)
        message_range = 4
    elif(dimension == 1):
        c_out = np.empty((0,1), np.float64)
        c = np.array([0.0]).astype(np.float64)
        message_range = 2
    else:
        print('invalid dimension: use 1,2 or 3')
        

    for i in np.ndindex(pc_input.shape[:-1]):
        # print('i', i[0], pc_input[i])

        # msg_val = message[i[0]]
        # print('msg val', msg_val)
        # bin_msg = bin(msg_val)[2:]
        
        #by anaylyzing the ouput of the format function, seems like for the values that can be represented by single or two bits like 0,1,2,3 the 0:02b and 0:01b use singe or double bits but for higher order bits.. it doesnt matter like 0:02b.format(8) or 0:01b.format(8) would still give '1000'
        #  
        if(dimension == 3):
            bin_msg = '{0:03b}'.format(msg_val)
        elif(dimension == 2):
            bin_msg = '{0:02b}'.format(msg_val)
        elif(dimension ==1):
            bin_msg = '{0:01b}'.format(msg_val)
        else:
            print('wrong dimension)')

        # print('message, bin message', msg_val, bin_msg)
        if(dimension == 3 or dimension == 2 or dimension == 1 ):
            if(bin_msg[0] == '0'):
                if(encode_type == 'dither'):
                    d0 = d0_estimator(resolution_delta, range_factor)
                else:
                    d0 = 0
                c[0] = dither_quantization_encode(pc_input[i][0], resolution_delta, d0)
            elif(bin_msg[0] == '1'):
                if(encode_type == 'dither'):
                    d1 = d1_estimator(resolution_delta, range_factor)
                else:
                    d1 = resolution_delta/2.0

                c[0] = dither_quantization_encode(pc_input[i][0], resolution_delta, d1)
            else:
                print('invalid bin data')

        if(dimension == 3 or dimension == 2):
            if(bin_msg[1] == '0'):
                if(encode_type == 'dither'):
                    d0 = d0_estimator(resolution_delta, range_factor)
                else:
                    d0 = 0
                c[1] = dither_quantization_encode(pc_input[i][1], resolution_delta, d0)
            elif(bin_msg[1] == '1'):
                if(encode_type == 'dither'):
                    d1 = d1_estimator(resolution_delta, range_factor)
                else:
                    d1 = resolution_delta/2.0

                c[1] = dither_quantization_encode(pc_input[i][1], resolution_delta, d1)
            else:
                print('invalid bin data')

        if(dimension == 3):
            if(bin_msg[2] == '0'):
                if(encode_type == 'dither'):
                    d0 = d0_estimator(resolution_delta, range_factor)
                else:
                    d0 = 0
                c[2] = dither_quantization_encode(pc_input[i][2], resolution_delta, d0)
            elif(bin_msg[2] == '1'):
                if(encode_type == 'dither'):
                    d1 = d1_estimator(resolution_delta, range_factor)
                else:
                    d1 = resolution_delta/2.0 
                c[2] = dither_quantization_encode(pc_input[i][2], resolution_delta, d1)
            else:
                print('invalid bin data')
                
        count+=1
        if(count == rate):
            msg_val += 1
            msg_val = msg_val % message_range
            # print('count, msg_val', count, msg_val)
            count = 0

        #print('count,message', count, message)
        c_out = np.append(c_out, [c], axis = 0)
        #print('c_out', c_out)
        
    return(np.array(c_out).astype(np.float64))


def d0_estimator(step, range_factor):
    d0 = Q0_randomGen(step, range_factor)
    # print('d0', d0)
    return(d0)

def d1_estimator(step, range_factor):
    d0 = d0_estimator(step, range_factor)
    if(d0 < 0):
        d1 = d0 + (step/2.0)
    else:
        d1 = d0 - (step/2.0)
    # print('d1', d1)
    return(d1)

def random_generator(range):
    #randin is a uniform distribution function
    value = randint(0,range)
    # print('rand val', value)
    return(value)

def Q0_randomGen_old(step):
    Q0_vec = [-(step/2.0), -(step/4.0), -(step/8.0), (step/8.0), (step/4.0), (step/2.0)]
    #generate a random index
    index = random_generator(5)
    #pick the value from the array Q0_vec
    val = Q0_vec[index]
    return(val)

def Q0_randomGen(step, step_range):
    a = -(step/float(step_range))
    b = (step/float(step_range))
    val = np.random.uniform(low=a, high=b, size=(1,))
    return(val)

def dither_quantization_encode(val, step, dm):
    # print('val', val)
    val =  val + dm
    # print('val+dm', val)
    # quantized_value = (np.around(val/step)*step - dm).astype(np.float64)
    quantized_value = (  (np.around(val/step)*step).astype(np.float64) - dm).astype(np.float64)
    return(quantized_value)
    

# def qim_decode_dither(pc_input, resolution_delta):

        
#         message_decoded = []
#         #print('c_dist shape', c_dist)
#         # print('message', message)            
        
#         for i in np.ndindex(pc_input.shape[:-1]):
#             c_dist = []
            
#             print('*********************')    
#             print('i',i)
            
#             c = np.array([0.0,0.0,0.0]).astype(np.float64)
#             c[0] = pc_input[i][0]
#             c[1] = pc_input[i][1]
#             c[2] = pc_input[i][2]
#             print ('C', c)

#             d = np.array([0.0,0.0,0.0]).astype(np.float64)
#             d0 = d0_estimator(resolution_delta)
#             d[0] = dither_quantization_encode(pc_input[i][0], resolution_delta, d0)

#             d0 = d0_estimator(resolution_delta)
#             d[1] = dither_quantization_encode(pc_input[i][1], resolution_delta, d0)

#             d0 = d0_estimator(resolution_delta)
#             d[2] = dither_quantization_encode(pc_input[i][2], resolution_delta, d0)

#             print ('D', d)

#             f = np.array([0.0,0.0,0.0]).astype(np.float64)  
#             d1 = d1_estimator(resolution_delta)
#             f[0] = dither_quantization_encode(pc_input[i][0], resolution_delta, d1)

#             d1 = d1_estimator(resolution_delta)
#             f[1] = dither_quantization_encode(pc_input[i][1], resolution_delta, d1)

#             d1 = d1_estimator(resolution_delta)
#             f[2] = dither_quantization_encode(pc_input[i][2], resolution_delta, d1)

#             print('F', f)

#             print('-------------')    

#             message = []
#             if(abs(c[0]-d[0]) < abs(c[0]-f[0])):
#                 message.append(0)
#             else:
#                 message.append(1)
#             if(abs(c[1]-d[1]) < abs(c[1]-f[1])):
#                 message.append(0)
#             else:
#                 message.append(1)
#             if(abs(c[2]-d[2]) < abs(c[2]-f[2])):
#                 message.append(0)
#             else:
#                 message.append(1)
            



#             message_decoded.append(message)

#         # print('message decoded', message_decoded)
#         return(message_decoded)

def correlationCoefficient(X, Y, n) : 
    sum_X = 0
    sum_Y = 0
    sum_XY = 0
    squareSum_X = 0
    squareSum_Y = 0
      
      
    i = 0
    while i < n : 
        # sum of elements of array X. 
        sum_X = sum_X + X[i] 
          
        # sum of elements of array Y. 
        sum_Y = sum_Y + Y[i] 
          
        # sum of X[i] * Y[i]. 
        sum_XY = sum_XY + X[i] * Y[i] 
          
        # sum of square of array elements. 
        squareSum_X = squareSum_X + X[i] * X[i] 
        squareSum_Y = squareSum_Y + Y[i] * Y[i] 
          
        i = i + 1
       
    # use formula for calculating correlation  
    # coefficient. 
    corr = (float)(n * sum_XY - sum_X * sum_Y)/(float)(math.sqrt((n * squareSum_X - sum_X * sum_X)* (n * squareSum_Y - sum_Y * sum_Y))) 
    return corr


def get_tamperedindices_dither_threebits(decoded_codebook, rate, length, threshold):
    #generate the encoded code book for the same rate
    
    error_counter = 0
    suspect_indices = []
    #generates dummy encoded bit stream with the rate upto the length specified by the 'length' parameter
    encoded_CB = encode_bitstream(rate, length)
    # print('encoded_cb', encoded_CB, encoded_CB.shape)

    #make sure the lengths of encoded and decoded code books are same
    if(len(decoded_codebook) == len(encoded_CB)): 
        # print('encoded CB and decoded CB lengths match')
        for index in range(len(decoded_codebook)):
            changed_indices = 0
            changed_indices = np.where(decoded_codebook[index] != encoded_CB[index])
            
                
            if(len(changed_indices[0])):
                # print('length of changed indices', len(changed_indices[0]))
                # print('encoded CB val', encoded_CB[index])
                # print('decoded CB val', decoded_codebook[index])
            
                if(threshold == 'full'):
                    #since the index starts from 0, to compensate for the first set of points we increase the index by 1 when trying to get the actual index value in suspect indices 
                    suspect_indices.append((index+1)*rate)    
                    for i in range(len(changed_indices[0])):
                        # increment the error counter
                        error_counter += 1
        
                elif(threshold == 'half'):
                    if(len(changed_indices[0]) > 1):
                        suspect_indices.append((index+1)*rate)    
                        for i in range(len(changed_indices[0])):
                        # increment the error counter
                            error_counter += 1
                else:
                    print('invalid threshold option')

    else:
       print('lengths do not match: DecodeCB, EncodeCB',  len(decoded_codebook), len(encoded_CB))    
    
    suspect_indices = np.array(suspect_indices).astype(np.int32)-1 #this substracting of one is a hack to fix the value of index crossing the size of the point cloud

    # lensuspect = len(suspect_indices) 
    # if(lensuspect == 0):
    #     lensuspect = 0.1
    # print('error counter', error_counter)
    #ber percentage
    error_rate =  (error_counter/(3.0*len(decoded_codebook)))*100

    return suspect_indices, error_rate 


def encode_bitstream(rate, pc_length):
    
    count = 0
    msg_val = 0
    c_out = np.empty((0,3), np.int8)
    c = np.array([0,0,0]) 
    
    for i in range(0,pc_length):
        
        bin_msg = '{0:03b}'.format(msg_val)
        # print('message, bin message', msg_val, bin_msg)
        c[0] = int(bin_msg[0])
        c[1] = int(bin_msg[1])
        c[2] = int(bin_msg[2])
        
        ## uncomment this line if you want the entire point cloud
        # c_out = np.append(c_out, [c], axis = 0)
        #print('c_out', c_out)

        count+=1
        if(count == rate):
            msg_val += 1
            msg_val = msg_val % 8
            # print('count, msg_val', count, msg_val)
            count = 0
            #here we just put the average (same as prediction)
            c_out = np.append(c_out, [c], axis = 0)
            

    return(np.array(c_out).astype(np.int8))

def qim_decode_generic(pc_input, resolution_delta, rate, range_factor, dimension, encode_type):

        
        message_decoded = []
        #print('c_dist shape', c_dist)
        # print('message', message)
        #             
        msg_val = 0

        if(encode_type == 'normal'):
            rate = 1
            range_factor = 1

        if(dimension == 3):
            Q0_diff_sum = np.array([0.0,0.0,0.0]).astype(np.float64)
            Q1_diff_sum = np.array([0.0,0.0,0.0]).astype(np.float64)
            message_range = 8
        elif (dimension == 2 ):
            Q0_diff_sum = np.array([0.0,0.0]).astype(np.float64)
            Q1_diff_sum = np.array([0.0,0.0]).astype(np.float64)
            message_range = 4
        elif (dimension == 1 ):
            Q0_diff_sum = np.array([0.0]).astype(np.float64)
            Q1_diff_sum = np.array([0.0]).astype(np.float64)
            message_range = 2

        count = 0
        for i in np.ndindex(pc_input.shape[:-1]): 
            # print('*********************')    
            # print('i',i)
            

            # c = np.array([0.0,0.0,0.0]).astype(np.float64)
            
            # # input row
            # c[0] = pc_input[i][0]
            # c[1] = pc_input[i][1]
            # c[2] = pc_input[i][2]
            # print ('C', c)

            #Q0 decoder
            
            # d = np.array([0.0,0.0,0.0]).astype(np.float64)
            if(dimension == 3):
                d = np.array([0.0,0.0,0.0]).astype(np.float64)
            elif(dimension == 2):
                d = np.array([0.0,0.0]).astype(np.float64)
            elif(dimension == 1):
                d = np.array([0.0]).astype(np.float64)
            else:
                print('invalid dimension: use 1,2 or 3')



            if(dimension == 3 or dimension == 2 or dimension == 1):
                if(encode_type == 'dither'):
                    d0 = d0_estimator(resolution_delta, range_factor)
                else:
                    d0 = 0

                d[0] = dither_quantization_encode(pc_input[i][0], resolution_delta, d0)

            if(dimension == 3 or dimension == 2):

                if(encode_type == 'dither'):
                    d0 = d0_estimator(resolution_delta, range_factor)
                else:
                    d0 = 0

                # d0 = d0_estimator(resolution_delta, range_factor)
                d[1] = dither_quantization_encode(pc_input[i][1], resolution_delta, d0)

            if(dimension == 3 ):
                # d0 = d0_estimator(resolution_delta, range_factor)
                if(encode_type == 'dither'):
                    d0 = d0_estimator(resolution_delta, range_factor)
                else:
                    d0 = 0
                d[2] = dither_quantization_encode(pc_input[i][2], resolution_delta, d0)

            # print ('Q0 decoder', d)
            

            # f = np.array([0.0,0.0,0.0]).astype(np.float64)
            
                
            if(dimension == 3):
                f = np.array([0.0,0.0,0.0]).astype(np.float64)
            elif(dimension == 2):
                f = np.array([0.0,0.0]).astype(np.float64)
            elif(dimension == 1):
                f = np.array([0.0]).astype(np.float64)
            else:
                print('invalid dimension: use 1,2 or 3')
            
            
            #Q1 decoder
            if(dimension == 3 or dimension == 2 or dimension == 1):  
                # d1 = d1_estimator(resolution_delta, range_factor)
                
                if(encode_type == 'dither'):
                    d1 = d1_estimator(resolution_delta, range_factor)
                else:
                    d1 = resolution_delta/2.0
                
                f[0] = dither_quantization_encode(pc_input[i][0], resolution_delta, d1)

            if(dimension == 3 or dimension == 2 ):
                if(encode_type == 'dither'):
                    d1 = d1_estimator(resolution_delta, range_factor)
                else:
                    d1 = resolution_delta/2.0
                # d1 = d1_estimator(resolution_delta, range_factor)
                f[1] = dither_quantization_encode(pc_input[i][1], resolution_delta, d1)

            if(dimension == 3 ):
                # d1 = d1_estimator(resolution_delta, range_factor)
                if(encode_type == 'dither'):
                    d1 = d1_estimator(resolution_delta, range_factor)
                else:
                    d1 = resolution_delta/2.0                
                
                f[2] = dither_quantization_encode(pc_input[i][2], resolution_delta, d1)

            # print('Q1 decoder', f)
            

            #accumulate sum until the rate bytes
            count+=1

            if(count == rate):
                
                msg_val += 1
                msg_val = msg_val % message_range
                # print('count, msg_val', count, msg_val)
                count = 0
                
                
                # print('-------------')    

                message = []
                #pick the value 
                if(dimension == 3 or dimension == 2 or dimension == 1):
                    if(Q0_diff_sum[0] < Q1_diff_sum[0]):
                        message.append(0)
                    else:
                        message.append(1)
                if(dimension == 3 or dimension == 2):
                    if(Q0_diff_sum[1] < Q1_diff_sum[1]):
                        message.append(0)
                    else:
                        message.append(1)
                if(dimension == 3):
                    if(Q0_diff_sum[2] < Q1_diff_sum[2]):
                        message.append(0)
                    else:
                        message.append(1)

                message_decoded.append(message)
                
                #reset the accumulators
                # Q0_diff_sum = np.array([0.0,0.0,0.0]).astype(np.float64)
                # Q1_diff_sum = np.array([0.0,0.0,0.0]).astype(np.float64)
                if(dimension == 3):
                    Q0_diff_sum = np.array([0.0,0.0,0.0]).astype(np.float64)
                    Q1_diff_sum = np.array([0.0,0.0,0.0]).astype(np.float64)
                elif (dimension == 2 ):
                    Q0_diff_sum = np.array([0.0,0.0]).astype(np.float64)
                    Q1_diff_sum = np.array([0.0,0.0]).astype(np.float64)
                elif (dimension == 1 ):
                    Q0_diff_sum = np.array([0.0]).astype(np.float64)
                    Q1_diff_sum = np.array([0.0]).astype(np.float64)

            else: #add the difference values until the count meets the rate
                # Q0_diff_sum[0] = Q0_diff_sum[0] + abs(c[0]-d[0])
                # Q1_diff_sum[0] = Q1_diff_sum[0] + abs(c[0]-f[0])

                # Q0_diff_sum[1] = Q0_diff_sum[1] + abs(c[1]-d[1])
                # Q1_diff_sum[1] = Q1_diff_sum[1] + abs(c[1]-f[1])
                
                # Q0_diff_sum[2] = Q0_diff_sum[2] + abs(c[2]-d[2])
                # Q1_diff_sum[2] = Q1_diff_sum[2] + abs(c[2]-f[2])

                if(dimension == 3 or dimension == 2 or dimension == 1):
                    Q0_diff_sum[0] = Q0_diff_sum[0] + (c[0]-d[0])**2
                    Q1_diff_sum[0] = Q1_diff_sum[0] + (c[0]-f[0])**2
                if(dimension == 3 or dimension == 2 ):
                    Q0_diff_sum[1] = Q0_diff_sum[1] + (c[1]-d[1])**2
                    Q1_diff_sum[1] = Q1_diff_sum[1] + (c[1]-f[1])**2
                if(dimension == 3 ):
                    Q0_diff_sum[2] = Q0_diff_sum[2] + (c[2]-d[2])**2
                    Q1_diff_sum[2] = Q1_diff_sum[2] + (c[2]-f[2])**2

        # print('message decoded', message_decoded)
        return(message_decoded)

if __name__ == '__main__':
    seed(1)
    range_factor_local = 3.0

    # 1. encode the point cloud and extract the codebook
    block_size_local = 16
    delta = 0.1
    

    pc_input = pc_test.reshape(-1,3)

    # noise =  np.random.uniform(-delta/4, delta/4, 3*len(pc_input)).reshape(-1,3)
    # print ('noise', noise, noise.shape)
    noise = 0.0
    pc_input = pc_input + noise 


    length_pc = len(pc_input)
    print('point cloud shape and values', pc_input.shape, pc_input)
    
    #usage: def qim_encode_generic(pc_input, resolution_delta, rate, range_factor, encode_type, dimension, message_value):  
    pc_output_dither = qim_encode_generic(pc_input, delta, block_size_local, range_factor_local, 'normal', 3 , 0)
    print('dither: output point cloud shape and values', pc_output_dither.shape, pc_output_dither)
    
    # pc_output_old = qim_encode_old(pc_input, delta)
    # print('old: output point cloud shape and values', pc_output_old.shape, pc_output_old)
    
    #usage: def qim_decode_generic(pc_input, resolution_delta, rate, range_factor, dimension, encode_type):
    decode = qim_decode_generic(pc_output_dither, delta, block_size_local,range_factor_local, 3, 'normal')
    
    decode_ = np.asarray(decode).reshape(-1)
    print('decoded message', decode_.reshape(-1,3))

    sample_encode = encode_bitstream(block_size_local, length_pc)
    # print('sample encode', sample_encode)
    

    # encode = np.array([1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1])
    # encode_ = np.array([1,1,1, 1,1,1, 1,1,1, 0,1,1, 0,1,1, 1,0,0, 0,0,0, 0,0,0])
    encode_ = np.asarray(sample_encode).reshape(-1)
    print('encode', encode_.reshape(-1,3))



    lcs = np.correlate(encode_, decode_)
    print('lcs', lcs)
    print('norm lcs', lcs/float(len(encode_)))
    
    # corr_val = (np.correlate(encode, decode)[0]/len(encode)).astype(np.float64)

    # print('corr', corr_val ) #//len(encode))
    n = len(encode_)

  
    # Function call to correlationCoefficient. 
    print ('correlation {0:.6f}'.format(correlationCoefficient(encode_, decode_, n))) 

    corr_pear = pearsonr(encode_, decode_)[0]
    print('pearson corr ', corr_pear)

    wrong_indices, ber =  get_tamperedindices_dither_threebits(decode_.reshape(-1,3), block_size_local, len(pc_input), 'full')

    print('WI full', wrong_indices)
    print('ber full', ber)


    wrong_indices, ber =  get_tamperedindices_dither_threebits(decode_.reshape(-1,3), block_size_local, len(pc_input), 'half')

    print('WI half', wrong_indices)
    print('ber half', ber)

    # //len(encode)).astype(np.float32))
    # seed(1)
    # for i in range(0,20):
    #     print('Q0', Q0_randomGen(100))
    # print('-------------')
    # # seed(2)
    # for i in range(0,20):
    #     print('Q0', Q0_randomGen(100))