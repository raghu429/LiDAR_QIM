#!/usr/bin/env python
import numpy as np
from decimal import Decimal

#pc = np.array(  [[1.13,1.08, 1.02],
#                [1.03, 1.05, 1.04],
#                [1.09,1.01, 1.12],
#                [1.01, 1.07, 1.03]]).astype(float)
                
                
pc = np.array(  [[1.12, 0.04, 1.32],
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
[1.70,1.80, 1.90]]).astype(float)


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

x = (0,1.4)
y = (0,1.2)
z = (0,1.5)

resolution = 0.05

count = 0
print('length of pc', len(pc))

def qim_decode(pc):
    #we get one point in the point cloud.. need to figure out its quantization
    # We need to build logic to recognize the code based on the values of the correspoinding points
    # We can check if each coordinate is even or odd.. while encoding ... (note this logic would get thrown off
    #if there are zeros in the quantized values)
    # if all the values are even or odd then the code could be zero or 1
    # so we can essentially recognize  000 or 111 (all odds and all evens)
    # 011 or 100 ()
    # 101 or 010 ()
    # 001 or 110 ()
    # We could reduce the encoding to four points to eliminate the symmetry issue
    # we could start with 111, 011, 101, 110  can also be represented as 1,3,5,6    
    
    # if we start in the same pattern we should get the modulo eight on count and get the start 
    #then confirm with the other logic
    
    final_code = []
    cloud_row_read = []
    for j in range (pc.shape[0]):
        even_flag = []       
        even_flag = get_evenflag(pc[j])
        # for i in range(3):
        #     #print('index: value:', (i,j), pc[j][i])
        #     #extract decimal number
            
        #     number_dec = int(pc[j][i] / resolution)
        #     print('num dec', number_dec)
        #     if (number_dec % 2 == 0): #even
        #         even_flag.append(1)
        #     else:
        #         even_flag.append(0)

        #     # number_dec = int(str(pc[j][i]).split('.')[1][:2])
        #     # print('num dec', number_dec)
        #     # if (number_dec % 2 == 0): #even
        #     #     even_flag.append(1)
        #     # else:
        #     #     even_flag.append(0)
        final_code.append( get_decode_codebook(even_flag))
    cloud_row_read = ((pc - np.array([x[0],y[0],z[0]])) / resolution).astype(np.int32)
    # print('******cloud read', cloud_row_read.shape, cloud_row_read)
    
    return final_code, cloud_row_read

def get_evenflag(pc_row):
    final_code = []
    e_flag = [] 
    # print('pc input', pc_row)
    for i in range(3):
            #print('index: value:', (i,j), pc[j][i])
            #extract decimal number
            
            number_dec = int(pc_row[i] / resolution)
            # print('num dec', number_dec)
            if (number_dec % 2 == 0): #even
                e_flag.append(1)
            else:
                e_flag.append(0)
    # final_code.append( get_decode_codebook(e_flag) )
    # print('even flag', e_flag)
    return e_flag
    
def get_decode_codebook(pc_row):
    
    code = []    
    
    # if(pc_row == [0,0,0] or pc_row == [1,1,1]):
    #     code = [1,1,1]
    # elif(pc_row == [0,1,1] or pc_row == [1,0,0]):
    #     code =  [0,1,1]
    # elif(pc_row == [1,0,1] or pc_row == [0,1,0]):
    #     code = [1,0,1]
    # elif(pc_row == [0,0,1] or pc_row == [1,1,0]):
    #     code =  [1,1,0]

    code = pc_row
    #print('row and code', (pc_row), (code))
    return code


def qim_quantize_restricted(pc, count):
    cloud_row_clean = ((pc - np.array([x[0],y[0],z[0]])) / resolution).astype(np.int32)
    # print('clean', cloud_row_clean)
    cloud_row = []
    cloud_row =  cloud_row_clean
    cloud_to_compare = []
    
    cloud_to_compare = np.array(get_evenflag(pc))
    # print('cloud to compare', cloud_to_compare)

    code_book = [0,0,0]
    
    # if(np.mod(count, 8) == 7 or np.mod(count, 8) == 0): # [111]  7 or 0
    #     if(cloud_to_compare == [1,1,1] or cloud_to_compare == [0,0,0]):
    #         print('condition exisits [1,1,1]')
    #         code_book = [1,1,1]
    #     else:
    #         cloud_row = cloud_row + 1    
    #         print('condition [111]', (count), cloud_row)
    #         code_book = [1,1,1]   
    # elif(np.mod(count, 8) == 3 or np.mod(count, 8) == 4): # [011] 4 or 3
    #     if(cloud_to_compare == [0,1,1] or cloud_to_compare == [1,0,0]):
    #         print('condition exisits [0,1,1]')
    #         code_book = [0,1,1]
    #     else:
    #         cloud_row[1] = cloud_row[1] + 1
    #         cloud_row[2] = cloud_row[2] + 1        
    #         code_book = [0,1,1]
    #         print('condition [011]', (count), cloud_row)
    # elif(np.mod(count, 8) == 5 or np.mod(count, 8) == 2): # [101] 5 or 2
    #     if(cloud_to_compare == [1,0,1] or cloud_to_compare == [0,1,0]):
    #         print('condition exisits [1,0,1]')
    #         code_book = [1,0,1]code_book = [1,1,1]
    #     else:
    #         cloud_row[0] = cloud_row[0] + 1
    #         cloud_row[2] = cloud_row[2] + 1
    #         code_book = [1,0,1]
    #         print('condition [101]', (count), cloud_row)
    # elif(np.mod(count, 8) == 6 or np.mod(count, 8) == 1): # [110] 6 or 1
    #     if(cloud_to_compare == [1,1,0] or cloud_to_compare == [0,0,1]):
    #         print('condition exisits [1,1,0]')
    #         code_book = [1,1,0]
    #     else:
    #         cloud_row[0] = cloud_row[0] + 1
    #         cloud_row[1] = cloud_row[1] + 1
    #         code_book = [1,1,0]
    #         print('condition [110]', (count), cloud_row)

    encode_cb_7 = np.array([1,1,1])
    encode_cb_6 = np.array([1,1,0])
    encode_cb_5 = np.array([1,0,1])
    encode_cb_3 = np.array([0,1,1])

    
    print('cloud compare', cloud_to_compare)
    if(np.mod(count, 4) == 0): # 0,4,8 ...
        # print('encodecb_7', encode_cb_7)
        if( (cloud_to_compare == encode_cb_7).all()):
            print('**********condition exisits [1,1,1]')
        else:
            changed_indices = np.where(cloud_to_compare != encode_cb_7)
            # print('changed indices',changed_indices )
            for i in range(len(changed_indices[0])):
                if(encode_cb_7[changed_indices[0][i]] == 1):
                    cloud_row[changed_indices[0][i]] = cloud_row[changed_indices[0][i]] + 1
                else:
                    cloud_row[changed_indices[0][i]] = cloud_row[changed_indices[0][i]] - 1

            # print('condition [111]', (count), cloud_row)
        code_book = encode_cb_7 #[1,1,1]   
    
    elif(np.mod(count, 4) == 1): # 1,5,9 ...
        # print('encodecb_6', encode_cb_6)
        if( (cloud_to_compare == encode_cb_6).all()):
            print('**********condition exisits [1,1,0]')
        else:
            changed_indices = np.where(cloud_to_compare != encode_cb_6)
            # print('changed indices**', changed_indices)
            for i in range(len(changed_indices[0])):
                if(encode_cb_6[changed_indices[0][i]] == 1):
                    cloud_row[changed_indices[0][i]] = cloud_row[changed_indices[0][i]] + 1
                else:
                    cloud_row[changed_indices[0][i]] = cloud_row[changed_indices[0][i]] - 1

            # print('condition [110]', (count), cloud_row)
        code_book = encode_cb_6 #[1,1,0]

    elif(np.mod(count, 4) == 2): # 2,6,10 ...
        # print('encodecb_5', encode_cb_5)
        if( (cloud_to_compare == encode_cb_5).all()):
            print('**********condition exisits [1,0,1]')
        else:
            changed_indices = np.where(cloud_to_compare != encode_cb_5)
            print('change indices & length', changed_indices, len(changed_indices[0]))
            for i in range(len(changed_indices[0])):
                if(encode_cb_5[changed_indices[0][i]] == 1):
                    cloud_row[changed_indices[0][i]] = cloud_row[changed_indices[0][i]] + 1
                else:
                    cloud_row[changed_indices[0][i]] = cloud_row[changed_indices[0][i]] - 1

            # print('condition [101]', (count), cloud_row)
        code_book = encode_cb_5 #[1,0,1]

    elif(np.mod(count, 4) == 3): # 3,7,11 ...
        # print('encodecb_3', encode_cb_3)
        if( (cloud_to_compare == encode_cb_3).all()):
            print('**********condition exisits [1,1,0]')
        else:
            changed_indices = np.where(cloud_to_compare != encode_cb_3)
            for i in range(len(changed_indices[0])):
                if(encode_cb_3[changed_indices[0][i]] == 1):
                    cloud_row[changed_indices[0][i]] = cloud_row[changed_indices[0][i]] + 1
                else:
                    cloud_row[changed_indices[0][i]] = cloud_row[changed_indices[0][i]] - 1

            # print('condition [011]', (count), cloud_row)
        code_book = encode_cb_3 #[0,1,1]

    #get the updated numpyh representation, updated point cloud and the codebook from data
    cloud_row_numpyrep = cloud_row
    cloud_row_pcvalues = cloud_row_numpyrep*resolution + np.array([x[0],y[0],z[0]])
    codebook_decoded = np.array(get_evenflag(cloud_row_pcvalues))
    codebook_basedoncount = code_book

    return cloud_row_numpyrep, codebook_basedoncount, cloud_row_pcvalues, codebook_decoded


def compare_codebooks(encode_cb, decode_cb):
    for i in range(encode_cb.shape[0]):
        if((encode_cb[i] == decode_cb[i]).all()):
            print(i)
        

# def qim_quantize(pc, count):
#     cloud_row_clean = ((pc - np.array([x[0],y[0],z[0]])) / resolution).astype(np.int32)
#     print('cloud row vlues', cloud_row_clean)
#     cloud_row =  cloud_row_clean
    
#     code_book = [0,0,0]
    
#     if(np.mod(count, 8) == 0):    
#         print('cloud row modified', cloud_row)
#         code_book = [0,0,0]   
#     elif(np.mod(count, 8) == 1):
#         cloud_row[2] = cloud_row_clean[2] + 1
#         code_book = [0,0,1]
#         print('cloud row modified', cloud_row)
#     elif(np.mod(count, 8) == 2):
#         cloud_row[1] = cloud_row_clean[1] + 1
#         code_book = [0,1,0]
#         print('cloud row modified', cloud_row)
#     elif(np.mod(count, 8) == 3):
#         cloud_row[1] = cloud_row_clean[1] + 1
#         cloud_row[2] = cloud_row_clean[2] + 1
#         code_book = [0,1,1]
#         print('cloud row modified', cloud_row)
#     elif(np.mod(count, 8) == 4):
#         cloud_row[0] = cloud_row_clean[0] + 1
#         code_book = [1,0,0]
#         print('cloud row modified', cloud_row)
#     elif(np.mod(count, 8) == 5):
#         cloud_row[0] = cloud_row_clean[0] + 1
#         cloud_row[2] = cloud_row_clean[2] + 1
#         code_book = [1,0,1]
#         print('cloud row modified', cloud_row)
#     elif(np.mod(count, 8) == 6):
#         cloud_row[0] = cloud_row_clean[0] + 1
#         cloud_row[1] = cloud_row_clean[1] + 1
#         code_book = [1,1,0]
#         print('cloud row modified', cloud_row)
#     elif(np.mod(count, 8) == 7):
#         cloud_row = cloud_row + 1
#         code_book = [1,1,1]
#         print('cloud row modified', cloud_row)
#     return cloud_row, code_book
    
pc_row_count = 0

#steps
quant_encoded = []
cbook = []
cbook_decoded = []
pc_values_decoded = []


# 1. encode the point cloud and extract the codebook

print('poitn cloud', pc_odd)
c = ((pc_odd - np.array([x[0],y[0],z[0]])) / resolution).astype(np.int32)
print('quantized input pc numpy representation', c)

quantized_pc  = c*resolution + np.array([x[0],y[0],z[0]])
print('quantized input pc values', quantized_pc) 


while pc_row_count < len(pc):
    quant_output, cbook_local, pc_values, cbook_from_pcvalues = qim_quantize_restricted(pc_odd[pc_row_count, :], pc_row_count)
    quant_encoded.append(quant_output)
    cbook.append(cbook_local)
    pc_values_decoded.append(pc_values)
    cbook_decoded.append(cbook_from_pcvalues)

    pc_row_count += 1
    if(pc_row_count > len(pc_odd)):
        print('something wrong.. exceeded length of pc')

encoded_pc = np.array([quant_encoded]).reshape(-1,3)
print('encoded pc numpy representation', encoded_pc)

# 2. Get the PC from the quatized values 
encoded_quantized_pc  = encoded_pc*resolution + np.array([x[0],y[0],z[0]])
print('encoded pc values', encoded_quantized_pc) 
# d = ((pc_odd - np.array([x[0],y[0],z[0]])) / resolution).astype(np.int32)

#3. decoded pc values and code book
print('decoded pc values',np.array([pc_values_decoded]).reshape(-1,3))
print('decoded code book',np.array([cbook_decoded]).reshape(-1,3))

# Decode the point cloud and get the code book
e, f = qim_decode(encoded_quantized_pc)
print('decoded pc numpy representation', np.array([f]).reshape(-1,3))

# ##c = a[:][logic_bound2]
# print('c', c.shape, c)
# print('d', c.shape, d)
decoded_codebook = np.array([e]).reshape(-1,3)
encoded_codebook = np.array([cbook]).reshape(-1,3)

# print('decoded_codebook', decoded_codebook)
print('encoded_codebook', encoded_codebook)

compare_codebooks(encoded_codebook, decoded_codebook)

