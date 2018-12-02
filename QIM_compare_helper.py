#!/usr/bin/env python
import numpy as np


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


decoded_code = np.array([   [0,0,0],
                            [0,0,1],
                            [0,1,0],

                            [0,0,0],
                            [0,0,1],
                            [0,1,0],
                            [0,1,1],
                            [1,0,0],
                            [1,0,1],
                            [1,1,0],
                            [1,1,1],
                            
                            [0,0,0],
                            [0,0,1],
                            [0,1,0],
                            [0,1,1],
                            [1,0,0],
                            [1,0,1],

                            [0,0,0],
                            [0,0,1],
                            [0,1,0],
                            [0,1,1],
                            [1,0,0],
                            [1,0,1],
                            [1,1,0],
                            [1,1,1],

                            [0,0,0],
                            [0,0,1],
                            [0,1,0],
                            [0,1,1],
                            [1,0,0],
                            [1,0,1],
                            [1,1,0],
                            [1,1,1],

                            [1,0,1],
                            [1,1,0],
                            [1,1,1],

                            [0,0,0],
                            [0,0,1],
                            [0,1,0],
                            [0,1,1],
                            [1,0,0],
                            [1,0,1],
                            [1,1,0],
                            [1,1,1],
                            [0,0,0],
                            
                            [0,0,0],
                            [0,0,1],
                            [1,1,1],

                            [0,0,1],
                            [0,1,0],
                            [0,1,1],
                            [1,0,0],
                            [1,0,1],
                            [1,1,0],
                            [1,1,1],

                            [0,0,0],
                            [0,0,1],
                            [0,1,0],
                            [0,1,1],
                            [1,0,0],
                            [1,0,1],
                            [1,1,0],
                            [1,1,1],
                            
                            [0,0,0]

                            ])
 
#cb_2 = np.array([[1, 1, 0],
# [0, 1, 0],
# [0, 0, 1],
# [0, 0, 0],
# [0, 0, 0],
# [0, 0, 1],
# [1, 1, 1],
# [1, 1, 0],
# [1, 0, 1],
# [1, 1, 0],
# [0, 0, 1],
# [0, 0, 0],
# [0, 1, 0]])



# def compare_pc(mod_block, base_cloud, block_move_size, threshold_val):
    
#     closeness = 0
#     total_count = 0
#     block_count = 0
    
#     print('mod block', mod_block)
#     while (block_count <= base_cloud.shape[0]-block_move_size):
#         base_block = base_cloud[block_count:block_count+block_move_size, :]
#         print('count  & base block', block_count, base_block)    
        
#         #compare the chunks
#         if (base_block.shape[0] == mod_block.shape[0]):
#             #flatten the blocks
#             base_block_flat = base_block.reshape(-1)
#             mod_block_flat = mod_block.reshape(-1)
#             print('base_block_flat:', base_block_flat)
#             print('mod_block_flat:', mod_block_flat)
            
                
#             for i in range(len(base_block_flat)):
#                 if (base_block_flat[i] == mod_block_flat[i]):
#                     closeness += 1
#                 total_count += 1
#         else:
#             print('blocks are of not the same size')
        
#         print('*****************iteration, closeness, totalcount', block_count, \
#         closeness, total_count)
        
#         if(float(closeness/total_count) > threshold_val):
#             return 1
#         else:
#             closeness = 0
#             total_count = 0
#             block_count += block_move_size
    
#     return 0


# def calculate_diff(base_pc, mod_pc, block_move_size, threshold_val):
#     block_count = 0
#     suspect_indices = []    
    
#     while (block_count <= mod_pc.shape[0]-block_move_size):
#         mod_block = mod_pc[block_count:block_count+block_move_size, :]
        
#         if(compare_pc(mod_block, base_pc[:], block_move_size, threshold_val)):
#             print('threshold reached')
#         else:
#             #mark the indices
#             suspect_indices.append(block_count)
#             print('suspect indicx list:', suspect_indices)
#             print('No match')
    
        
#         #advance the index
#         block_count  += block_move_size
#     if(len(suspect_indices) == 0):
#         print('clouds match')
#     else:
#         print('ekkado bokka vundi ra chusko')
#         print('suspect indices final', suspect_indices)




# def Get_tamperindices_pointbased(decoded_cb):
    
#     encode_cb_0 = np.array([0,0,0])
#     encode_cb_1 = np.array([0,0,1])
#     encode_cb_2 = np.array([0,1,0])
#     encode_cb_3 = np.array([0,1,1])

#     encode_cb_4 = np.array([1,0,0])
#     encode_cb_5 = np.array([1,0,1])
#     encode_cb_6 = np.array([1,1,0])
#     encode_cb_7 = np.array([1,1,1])
    
#     pc_row_count = 0
#     suspect_indices = []

#     # suspect_indices.append(block_count)
#     while pc_row_count < len(decoded_cb):
#         cloud_to_compare = []
#         cloud_to_compare = decoded_cb[pc_row_count, :]
        
#         if(np.mod(pc_row_count, 8) == 0): # 0,8, 16...
#             # print('encode_0', encode_cb_0)
#             if( (cloud_to_compare == encode_cb_0).all()):
#                 pass
#             else:
#                 suspect_indices.append(pc_row_count)

#         elif(np.mod(pc_row_count, 8) == 1): # 1,9, ...
#             # print('encodecb_1', encode_cb_1)
#             if( (cloud_to_compare == encode_cb_1).all()):
#                 pass
#             else:
#                 suspect_indices.append(pc_row_count)

#         elif(np.mod(pc_row_count, 8) == 2): # 2,10 ...
#             # print('encodecb_2', encode_cb_2)
#             if( (cloud_to_compare == encode_cb_2).all()):
#                 pass
#             else:
#                suspect_indices.append(pc_row_count)

#         elif(np.mod(pc_row_count, 8) == 3): # 3,11 ...
#             # print('encodecb_3', encode_cb_3)
#             if( (cloud_to_compare == encode_cb_3).all()):
#                 pass
#             else:
#                 suspect_indices.append(pc_row_count)
        
#         elif(np.mod(pc_row_count, 8) == 4): # 4,12 ...
#             # print('encodecb_4', encode_cb_4)
#             if( (cloud_to_compare == encode_cb_4).all()):
#                 pass
#             else:
#                suspect_indices.append(pc_row_count)
        
#         elif(np.mod(pc_row_count, 8) == 5): # 5,13 ...
#             # print('encodecb_5', encode_cb_5)
#             if( (cloud_to_compare == encode_cb_5).all()):
#                 pass
#             else:
#                 suspect_indices.append(pc_row_count)
        
#         elif(np.mod(pc_row_count, 8) == 6): # 6,14 ...
#             # print('encodecb_6', encode_cb_6)
#             if( (cloud_to_compare == encode_cb_6).all()):
#                 pass
#             else:
#                 suspect_indices.append(pc_row_count)
        
#         elif(np.mod(pc_row_count, 8) == 7): # 7,15 ...
#             # print('encodecb_7', encode_cb_7)
#             if( (cloud_to_compare == encode_cb_7).all()):
#                 pass
#             else:
#                 suspect_indices.append(pc_row_count)

#         pc_row_count = pc_row_count + 1
#         if(pc_row_count > len(decoded_cb)):
#             print('something wrong.. exceeded length of pc')

#     return suspect_indices


def get_tamperedpc_indices(input_indices, tolerance):
    list_length = input_indices.shape[0]

    if(list_length <= 2):
        return 0

    cluster_min_max = []

    for i in range(list_length):
        if (i == 0):
            cluster_min_max.append(input_indices[i])
        else:
            if(i== (list_length-1)):
                cluster_min_max.append(input_indices[i])

            if( input_indices[i] < input_indices[i-1] + tolerance):
                pass
            else:
                cluster_min_max.append(input_indices[i-1])
                cluster_min_max.append(input_indices[i])
    # return cluster_min_max
    return (np.array([cluster_min_max]).reshape(-1,2) )



def Get_tamperindices_blockbased(decoded_code):
    template_code = np.array([[0,0,0],
                        [0,0,1],
                        [0,1,0],
                        [0,1,1],
                        [1,0,0],
                        [1,0,1],
                        [1,1,0],
                        [1,1,1]])

    suspect_indices = []

    # print('template code size', template_code.shape[0])
    
    block_count = 0
    block_move_size = template_code.shape[0]

    block_count_increment = block_move_size
    # block_traverse_size = decoded_code.shape[0] - block_count_increment
    

    while (block_count <= decoded_code.shape[0]):
        # print('block count, block_count_increment', block_count, block_count_increment)

        block_move_length = (block_count + block_move_size)

        if(block_move_length - decoded_code.shape[0] > block_move_size):
            block_move_size = 1
            block_move_length = (block_count + block_move_size)
            
        mod_block = decoded_code[block_count:block_move_length, :]
        
        if(np.array_equal(mod_block, template_code) == False):
            block_count_increment = 1
            #mark the indices
            suspect_indices.append(block_count)
            # print('suspect index list:', suspect_indices)
        else:
            block_count_increment = block_move_size

        #advance the index
        block_count  += block_count_increment
    
    if(len(suspect_indices) == 0):
        print('PC intact')
    else:
        print('PC compromised')
        suspect_indices = np.array(suspect_indices)
        # print('suspect indices final', suspect_indices)
        tampered_pc_indices = get_tamperedpc_indices(suspect_indices, 2)
        print('tampered cluster indices', tampered_pc_indices)


if __name__ == '__main__':

    threshold =  0.7
    block_size = 2

    # cb_1 = np.random.randint(2, size = [10, 3])
    #print('cb_1:', cb_1)
    # cb_2 = np.random.randint(2, size = [13, 3])
    #print('cb_2:', cb_2)


    # max_size = max(cb_1.shape[0],cb_2.shape[0] )

    # window_size = 2
    # print('max pc size', max_size)
    # iter_count = int(max_size/window_size)
    # print('iter count:', iter_count)

    Get_tamperindices_blockbased(decoded_code)
    
    # suspect_index = Get_tamperindices_pointbased(decoded_code)
    # print('culprit indices', np.array([suspect_index]))

    # calculate_diff(cb_1,cb_2,block_size,threshold)



        
        
        

