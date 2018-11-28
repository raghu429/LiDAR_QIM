#!/usr/bin/env python
import numpy as np
from decimal import Decimal

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

x = (0,1.4)
y = (0,1.2)
z = (0,1.5)

resolution = 0.05


def qim_decode(pc):
    # We can check if each coordinate is even or odd.. while encoding 
    final_code = []
    cloud_row_read = []
    
    for j in range (pc.shape[0]):
        row_even_flags = []       
        row_even_flags = get_evenflag(pc[j])
        final_code.append(row_even_flags)

    cloud_row_read = getQuantizedValues_from_pointCloud(pc, resolution, x,y,z)
    # print('******cloud read', cloud_row_read.shape, cloud_row_read)
    
    return final_code, cloud_row_read

def getPointCloud_from_quantizedValues(pc_encoded, resolution_in, x_in, y_in, z_in):
    return(pc_encoded*resolution_in + np.array([x_in[0],y_in[0],z_in[0]]))


def getQuantizedValues_from_pointCloud(pc, resolution_in, x_in,y_in,z_in):
    return(((pc - np.array([x_in[0],y_in[0],z_in[0]])) / resolution_in).astype(np.int32))


#checks if the quantized value is even or odd
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
    

def qim_quantize_restricted(pc_in):
    
    encode_cb_0 = np.array([0,0,0])
    encode_cb_1 = np.array([0,0,1])
    encode_cb_2 = np.array([0,1,0])
    encode_cb_3 = np.array([0,1,1])

    encode_cb_4 = np.array([1,0,0])
    encode_cb_5 = np.array([1,0,1])
    encode_cb_6 = np.array([1,1,0])
    encode_cb_7 = np.array([1,1,1])
    
    pc_row_count = 0

    quant_encoded = []
    cbook = []
    # cbook_decoded = []
    # pc_values_decoded = []
    
    while pc_row_count < len(pc_in):
        cloud_to_compare = []
        cloud_row = []

        pc = pc_in[pc_row_count, :]
        # cloud_row_clean = ((pc - np.array([x[0],y[0],z[0]])) / resolution).astype(np.int32)
        quantized_values_row = getQuantizedValues_from_pointCloud(pc, resolution, x,y,z)
        # print('clean', cloud_row_clean)
        cloud_row =  quantized_values_row
        

        cloud_to_compare = np.array(get_evenflag(pc))
        # print('cloud to compare', cloud_to_compare)

        #reset code book
        code_book = [0,0,0]
        # print('cloud compare', cloud_to_compare)

        if(np.mod(pc_row_count, 8) == 0): # 0,8, 16...
            # print('encode_0', encode_cb_0)
            if( (cloud_to_compare == encode_cb_0).all()):
                print('**********condition exisits [0,0,0]')
            else:
                changed_indices = np.where(cloud_to_compare != encode_cb_0)
                # print('changed indices',changed_indices )
                for i in range(len(changed_indices[0])):
                    if(encode_cb_0[changed_indices[0][i]] == 1):
                        cloud_row[changed_indices[0][i]] = cloud_row[changed_indices[0][i]] + 1
                    else:
                        cloud_row[changed_indices[0][i]] = cloud_row[changed_indices[0][i]] - 1

                # print('condition [000]', (count), cloud_row)
            code_book = encode_cb_0 #[0,0,0]   
        
        elif(np.mod(pc_row_count, 8) == 1): # 1,9, ...
            # print('encodecb_1', encode_cb_1)
            if( (cloud_to_compare == encode_cb_1).all()):
                print('**********condition exisits [0,0,1]')
            else:
                changed_indices = np.where(cloud_to_compare != encode_cb_1)
                # print('changed indices**', changed_indices)
                for i in range(len(changed_indices[0])):
                    if(encode_cb_6[changed_indices[0][i]] == 1):
                        cloud_row[changed_indices[0][i]] = cloud_row[changed_indices[0][i]] + 1
                    else:
                        cloud_row[changed_indices[0][i]] = cloud_row[changed_indices[0][i]] - 1

                # print('condition [001]', (count), cloud_row)
            code_book = encode_cb_1 #[0,0,1]

        elif(np.mod(pc_row_count, 8) == 2): # 2,10 ...
            # print('encodecb_2', encode_cb_2)
            if( (cloud_to_compare == encode_cb_2).all()):
                print('**********condition exisits [0,1,0]')
            else:
                changed_indices = np.where(cloud_to_compare != encode_cb_2)
                print('change indices & length', changed_indices, len(changed_indices[0]))
                for i in range(len(changed_indices[0])):
                    if(encode_cb_5[changed_indices[0][i]] == 1):
                        cloud_row[changed_indices[0][i]] = cloud_row[changed_indices[0][i]] + 1
                    else:
                        cloud_row[changed_indices[0][i]] = cloud_row[changed_indices[0][i]] - 1

                # print('condition [101]', (count), cloud_row)
            code_book = encode_cb_2 #[1,0,1]

        elif(np.mod(pc_row_count, 8) == 3): # 3,11 ...
            # print('encodecb_3', encode_cb_3)
            if( (cloud_to_compare == encode_cb_3).all()):
                print('**********condition exisits [0,1,1]')
            else:
                changed_indices = np.where(cloud_to_compare != encode_cb_3)
                for i in range(len(changed_indices[0])):
                    if(encode_cb_3[changed_indices[0][i]] == 1):
                        cloud_row[changed_indices[0][i]] = cloud_row[changed_indices[0][i]] + 1
                    else:
                        cloud_row[changed_indices[0][i]] = cloud_row[changed_indices[0][i]] - 1

                # print('condition [011]', (count), cloud_row)
            code_book = encode_cb_3 #[0,1,1]
        
        elif(np.mod(pc_row_count, 8) == 4): # 4,12 ...
            # print('encodecb_4', encode_cb_4)
            if( (cloud_to_compare == encode_cb_4).all()):
                print('**********condition exisits [1,0,0]')
            else:
                changed_indices = np.where(cloud_to_compare != encode_cb_4)
                for i in range(len(changed_indices[0])):
                    if(encode_cb_3[changed_indices[0][i]] == 1):
                        cloud_row[changed_indices[0][i]] = cloud_row[changed_indices[0][i]] + 1
                    else:
                        cloud_row[changed_indices[0][i]] = cloud_row[changed_indices[0][i]] - 1

                # print('condition [100]', (count), cloud_row)
            code_book = encode_cb_4 #[1,0,0]
        
        elif(np.mod(pc_row_count, 8) == 5): # 5,13 ...
            # print('encodecb_5', encode_cb_5)
            if( (cloud_to_compare == encode_cb_5).all()):
                print('**********condition exisits [1,0,1]')
            else:
                changed_indices = np.where(cloud_to_compare != encode_cb_5)
                for i in range(len(changed_indices[0])):
                    if(encode_cb_3[changed_indices[0][i]] == 1):
                        cloud_row[changed_indices[0][i]] = cloud_row[changed_indices[0][i]] + 1
                    else:
                        cloud_row[changed_indices[0][i]] = cloud_row[changed_indices[0][i]] - 1

                # print('condition [101]', (count), cloud_row)
            code_book = encode_cb_5 #[1,0,1]
        
        elif(np.mod(pc_row_count, 8) == 6): # 6,14 ...
            # print('encodecb_6', encode_cb_6)
            if( (cloud_to_compare == encode_cb_4).all()):
                print('**********condition exisits [1,1,0]')
            else:
                changed_indices = np.where(cloud_to_compare != encode_cb_6)
                for i in range(len(changed_indices[0])):
                    if(encode_cb_3[changed_indices[0][i]] == 1):
                        cloud_row[changed_indices[0][i]] = cloud_row[changed_indices[0][i]] + 1
                    else:
                        cloud_row[changed_indices[0][i]] = cloud_row[changed_indices[0][i]] - 1

                # print('condition [110]', (count), cloud_row)
            code_book = encode_cb_6 #[1,1,0]
        
        elif(np.mod(pc_row_count, 8) == 7): # 7,15 ...
            # print('encodecb_7', encode_cb_7)
            if( (cloud_to_compare == encode_cb_7).all()):
                print('**********condition exisits [1,1,1]')
            else:
                changed_indices = np.where(cloud_to_compare != encode_cb_7)
                for i in range(len(changed_indices[0])):
                    if(encode_cb_3[changed_indices[0][i]] == 1):
                        cloud_row[changed_indices[0][i]] = cloud_row[changed_indices[0][i]] + 1
                    else:
                        cloud_row[changed_indices[0][i]] = cloud_row[changed_indices[0][i]] - 1

                # print('condition [111]', (count), cloud_row)
            code_book = encode_cb_7 #[1,1,1]
        

        quant_encoded.append(cloud_row)
        cbook.append(code_book)
        # print('row# and code book', pc_row_count, code_book)

        pc_row_count += 1
        if(pc_row_count > len(pc_in)):
            print('something wrong.. exceeded length of pc')
        
        # pc_values_decoded.append(pc_values)
        # cbook_decoded.append(cbook_from_pcvalues)

        #get the updated numpyh representation, updated point cloud and the codebook from data
        # cloud_row_numpyrep = cloud_row
        # cloud_row_pcvalues = cloud_row_numpyrep*resolution + np.array([x[0],y[0],z[0]])
        # codebook_decoded = np.array(get_evenflag(cloud_row_pcvalues))
        # codebook_basedoncount = code_book

    return quant_encoded, cbook


def compare_codebooks(encode_cb, decode_cb):
    for i in range(encode_cb.shape[0]):
        if((encode_cb[i] == decode_cb[i]).all()):
            print(i)
        


if __name__ == '__main__':


    # 1. encode the point cloud and extract the codebook

    pc_input = pc_even
    
    print('poitn cloud', pc_input)
    c = ((pc_input - np.array([x[0],y[0],z[0]])) / resolution).astype(np.int32)
    print('quantized input pc numpy representation', c)

    quantized_pc  = c*resolution + np.array([x[0],y[0],z[0]])
    print('quantized input pc values', quantized_pc) 

    encoded_quantized_values, encoded_CB = qim_quantize_restricted(pc_input)
    encoded_pc = np.array([encoded_quantized_values]).reshape(-1,3)
    print('encoded pc numpy representation', encoded_pc)

    # 2. Get the PC from the quantized values 
    # encoded_quantized_pc  = encoded_pc*resolution + np.array([x[0],y[0],z[0]])
    encoded_quantized_pc  = getPointCloud_from_quantizedValues(encoded_pc, resolution, x,y,z)
    print('encoded pc values', encoded_quantized_pc) 
    # d = ((pc_input - np.array([x[0],y[0],z[0]])) / resolution).astype(np.int32)

    # 3. Get the decoded codebook and quantized values of decoded point cloud   
    # Decode the point cloud and get the code book
    decoded_CB, decoded_quantized_values = qim_decode(encoded_quantized_pc)
    print('decoded pc numpy representation', np.array([decoded_quantized_values]).reshape(-1,3))

    decoded_codebook = np.array([decoded_CB]).reshape(-1,3)
    encoded_codebook = np.array([encoded_CB]).reshape(-1,3)

    print('decoded_codebook', decoded_codebook)
    print('encoded_codebook', encoded_codebook)

    #compare codebooks
    compare_codebooks(encoded_codebook, decoded_codebook)

