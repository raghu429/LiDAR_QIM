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



def compare_pc(mod_block, base_cloud):
    
    closeness = 0
    total_count = 0
    block_count = 0
    
    print('mod block', mod_block)
    while (block_count <= base_cloud.shape[0]-block_move_size):
        base_block = base_cloud[block_count:block_count+block_move_size, :]
        print('count  & base block', block_count, base_block)    
        
        #compare the chunks
        if (base_block.shape[0] == mod_block.shape[0]):
            #flatten the blocks
            base_block_flat = base_block.reshape(-1)
            mod_block_flat = mod_block.reshape(-1)
            print('base_block_flat:', base_block_flat)
            print('mod_block_flat:', mod_block_flat)
            
                
            for i in range(len(base_block_flat)):
                if (base_block_flat[i] == mod_block_flat[i]):
                    closeness += 1
                total_count += 1
        else:
            print('blocks are of not the same size')
        
        print('*****************iteration, closeness, totalcount', block_count, \
        closeness, total_count)
        
        if(float(closeness/total_count) > threshold_val):
            return 1
        else:
            closeness = 0
            total_count = 0
            block_count += block_move_size
    
    return 0


def calculate_diff(base_pc, mod_pc):
    block_count = 0
    suspect_indices = []    
    
    while (block_count <= mod_pc.shape[0]-block_move_size):
        mod_block = mod_pc[block_count:block_count+block_move_size, :]
        
        if(compare_pc(mod_block, base_pc[:])):
            print('threshold reached')
        else:
            #mark the indices
            suspect_indices.append(block_count)
            print('suspect indicx list:', suspect_indices)
            print('No match')
    
        
        #advance the index
        block_count  += block_move_size
    if(len(suspect_indices) == 0):
        print('clouds match')
    else:
        print('ekkado bokka vundi ra chusko')
        print('suspect indices final', suspect_indices)




threshold_val =  0.7
block_move_size = 2

# cb_1 = np.random.randint(2, size = [10, 3])
#print('cb_1:', cb_1)
# cb_2 = np.random.randint(2, size = [13, 3])
#print('cb_2:', cb_2)


max_size = max(cb_1.shape[0],cb_2.shape[0] )

window_size = 2
print('max pc size', max_size)
iter_count = int(max_size/window_size)
print('iter count:', iter_count)

calculate_diff(cb_1,cb_2)



        
        
        

