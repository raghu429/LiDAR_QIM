#!/usr/bin/env python
import numpy as np
#import pcl
#import os
#import ntpath
#from helper_functions import *
import seaborn as sns
import matplotlib.pyplot as plt




#****************************************************************************************
# box plot test
sns.set(style="whitegrid")
tips = sns.load_dataset("tips")

print('tips shape', tips.shape)
print('tips', tips)
plt.figure()
ax = sns.boxplot(x=tips["total_bill"])
plt.show()

# fig, ax = plt.subplots()
# # y = [add1*100, add2*100, add3*100, add4*100]
# y = [Del1*100, Del2*100, Del3*100, Del4*100]
# # y = [clean1*100, clean2*100, clean3*100, clean4*100]

# pos = np.array(range(len(y)))+ 1
# bp = ax.boxplot(y, sym='k+', positions=pos, labels = ['a','b','c', 'd'])


# ax.set_xlabel('Noise Variance')
# ax.set_ylabel('Bit Error Rate %')

# plt.setp(bp['whiskers'], color='k', linestyle='-')
# plt.setp(bp['fliers'], markersize=3.0)
# plt.show()

# *****************************************************************************************


#*****************************************************************************************
#person corr coeff

def generate_correlation_map(x, y):
    """Correlate each n with each m.

    Parameters
    ----------
    x : np.array
      Shape N X T.

    y : np.array
      Shape M X T.

    Returns
    -------
    np.array
      N X M array in which each element is a correlation coefficient.

    """
    mu_x = x.mean(1)
    mu_y = y.mean(1)
    n = x.shape[1]
    if n != y.shape[1]:
        raise ValueError('x and y must ' +
                         'have the same number of timepoints.')
    s_x = x.std(1, ddof=n - 1)
    s_y = y.std(1, ddof=n - 1)
    cov = np.dot(x,
                 y.T) - n * np.dot(mu_x[:, np.newaxis],
                                  mu_y[np.newaxis, :])
    return cov / np.dot(s_x[:, np.newaxis], s_y[np.newaxis, :])

from scipy.stats import pearsonr

def test_generate_correlation_map():
    # x = np.random.rand(10, 10)
    # y = np.random.rand(20, 10)
    
    x = np.random.randint(2, size=12)
    y = np.random.randint(2, size=12)

    x = x.reshape(-1,3)
    y = y.reshape(-1,3)


    print('x', x.shape, x)
    print('y', y.shape, y)

    x = x.reshape(-1,1)
    y = y.reshape(-1,1)

    x = [1,4,6]
    y = [1,2,3]  

    x = np.array(x)
    y = np.array(y)

    print('x', x.shape, x)
    print('y', y.shape, y)



    pear_val = pearsonr(x,y)

    print('pear val', pear_val)
    
    x = x.reshape(-1,1)
    y = y.reshape(-1,1)

    print('x', x.shape, x)
    print('y', y.shape, y)



    pear_val = pearsonr(x,y)

    print('pear val', pear_val)
    
    
    # desired = np.empty((10, 20))
    # for n in range(x.shape[0]):
    #     for m in range(y.shape[0]):
    #         desired[n, m] = pearsonr(x[n, :], y[m, :])[0]
    
    # actual = generate_correlation_map(x, y)
    # print('actua shape', actual.shape)
    # print('desired shape', desired.shape)
    # np.testing.assert_array_almost_equal(actual, desired)

test_generate_correlation_map()



# #*****************************************************************************************

# #***********************************************************************************
 #logical bounds tests

a = np.arange(27)
a = a.reshape(-1,3)
# print('a', a.shape, a)
c = a*0.05
# print('c', c)
#
##copy a into some other variable to save it
#aa = a.astype(float)
#
#b = a+1
#print('b and shape', b.shape, b)
#
#logic_bound1 = np.array([False, False, False, True, True, False, False, True, False])
#logic_bound2 = np.array([True, True, True, True, True, False, False, False, True])
#print('len of logc bound', len(logic_bound1))
#
#b = aa[:][logic_bound2]
#print('b', b.shape, b)


#pc = np.array(  [[1.13,1.08, 1.02],
#                [1.03, 1.05, 1.04],
#                [1.09,1.01, 1.12],
#                [1.01, 1.07, 1.03]]).astype(float)
#
#x = (0,1.4)
#y = (0,1.2)
#z = (0,1.5)
#resolution = 0.05
#
#count = 0
#
#for i in range(0, len(pc)):
#    
#    for j in range(0,3):
#        
#        if(np.mod(cont, 8) == 0):
#            print('count, value', count, pc[i][j])
#    
#    
#    
#    
#    count += 1
#    
#c = ((pc - np.array([x[0],y[0],z[0]])) / resolution).astype(np.int32)
#
#    
###c = a[:][logic_bound2]
#print('c', c.shape, c)
#
#
#voxel = np.zeros(( int((x[1] - x[0]) / resolution), int((y[1] - y[0]) / resolution), int(round((z[1]-z[0]) / resolution)) ))
#
#print('voxel as created', voxel.shape)
#voxel[c[:,0], c[:,1], c[:,2]] = 24
#print('voxel modified', voxel.shape, voxel)
#
#super_logicalbound = np.zeros(a.shape[0])
#super_logicalbound = np.logical_or(super_logicalbound, logic_bound1)
#
#super_logicalbound = np.logical_or(super_logicalbound, logic_bound2)
#
#e = a[super_logicalbound]
#print('e', e.shape, e)
#
#logic_bound_no12 = np.logical_not(super_logicalbound)
#f = a[logic_bound_no12]
#print('f', f.shape, f)

# #*****************************************************************************************

# #******************************************************************************************
# # list element distance calculation tests

# a = np.arange(27)
# a = a.reshape(-1,3)
# print('a', a.shape, a)

# b = a+1
# print('b and shape', b.shape, b)

# dist_a_b = dist(a[1], b[1])
# print('dist', dist_a_b)

# if(a.shape[0] != b.shape[0]):
#   print('not equa')
# else:
#     print('equa')
# #********************************************************************************************



# #*******************************************************************************************
# #noise addition tests

# a = np.arange(27)
# a = a.reshape(-1,3)
# print('a', a.shape, a)

# b = a+1
# print('b and shape', b.shape, b)

# logic_bound1 = [False, False, False, True, True, False, False, True, False]
# logic_bound2 = [True, False, False, False, True, False, False, False, True]

# #copy a into some other variable to save it
# aa = a.astype(float)

# def get_noiseaddedcloud(pointcloud_input, logical_bound_in, sigma_in, mean_in, axis_in):
# #Make a zeros matrix with same size as the logical_bounds
#     dummy = np.zeros(pointcloud_input[logical_bound_in].shape)
#     #make gaussian noise to the z-axis of this dummy matrix
#     noise = np.random.normal(mean_in, sigma_in, dummy[:,axis_in].shape[0])
#     #Add gaussian noise to zero matrix
#     dummy[:,axis_in] = dummy[:,axis_in] + noise
#     #modify the point cloud with the noise
#     pointcloud_input[logical_bound_in] = pointcloud_input[logical_bound_in] + dummy

# # #modify aa
# # dummy = np.zeros(aa[logic_bound1].shape)
# # print dummy
# # print ('dummy[:,2] shape', dummy[:,2].shape[0])
# # noise = np.random.normal(0, 0.3, dummy[:,2].shape[0])
# # dummy[:,2] = dummy[:,2] + noise
# # print dummy
# # aa[logic_bound1] = aa[logic_bound1] + dummy    
# get_noiseaddedcloud(aa, logic_bound1, 0.3, 0, 2)
# get_noiseaddedcloud(aa, logic_bound2, 0.1, 0, 2)
# print('aa', aa.shape, aa)

# #************************************************************************************************


# #***********************************************************************************************
# #test to check the sigmalist
# logic_bound1 = [False, False, False, True, True, False, False, True, False]
# logic_bound2 = [True, False, False, False, True, False, False, False, True]
# min_limit = 0.1
# max_limit  = 0.5
# gap = (max_limit - min_limit)/len(logic_bound1)
# sigmalist = np.arange(min_limit,max_limit,gap)
# print(sigmalist)
# #**********************************************************************************************

# #*********************************************************************************************
# #argsort test
# x = np.array([ [0, 3, 2], [2, 2, 1], [5, 1, 3], [6,8,4] ]).astype(np.float32)
# print(x)
# y = np.argsort(x[:,0], axis = 0)
# print(y)


# #*********************************************************************************************

# #*********************************************************************************************
# #range test
# x  = np.arange(0,7)
# print('x', x, len(x))
# for i in range(7):
#     print(i)
# #*********************************************************************************************


#***********************************************************************************************

# #kd-tree test
# points_1 = np.array([[0, 0, 0], #0
#                      [1, 0, 0], #1
#                      [0, 1, 0], #2
#                      [1, 1, 0]], dtype=np.float32)

# points_2 = np.array([[0, 0, 0], #0
#                      [1, 0, 0], #1
#                      ], dtype=np.float32)

# # [5, 4, 3], #2
# #                      [0, 1, 0], #3
# #                      [1, 1, 0]
# # points_2 = np.array([   [1.1, 1, 0.5],
# #                         [0, 0, 0.2],
# #                         [1, 0, 0],
# #                         [0, 1, 0],
# #                         [5,4,3]     ], dtype=np.float32)

# # pc_1 = pcl.PointCloud()
# # pc_1.from_array(points_1)
# # pc_2 = pcl.PointCloud()
# # pc_2.from_array(points_2)
# # kd = pcl.KdTreeFLANN(pc_1)

# print('pc_1:')
# print(points_1)
# print('\npc_2:')
# print(points_2)
# print('\n')

# pc_1 = pcl.PointCloud(points_1)
# pc_2 = pcl.PointCloud(points_2)

# # kd = pc_1.make_kdtree_flann()

# # # find the single closest points to each point in point cloud 2
# # # (and the sqr distances)
# # indices, sqr_distances = kd.nearest_k_search_for_cloud(pc_2, 1)

# # for i in range(pc_1.size):
# #     print('pc2: %d, pc1: %d, distance %f' % ( i, indices[i, 0], sqr_distances[i, 0]) )


# kd = pc_2.make_kdtree_flann()

# # find the single closest points to each point in point cloud 2
# # (and the sqr distances)
# indices, sqr_distances = kd.nearest_k_search_for_cloud(pc_1, 1)

# for i in range(pc_1.size):
#     print('pc1: %d, pc2: %d, distance %f' % ( i, indices[i, 0], sqr_distances[i, 0]) )

#***********************************************************************************************

#***********************************************************************************************
#python list call by reference

# lista = [1,2,3]

# def funa(somelist):
#     print somelist
#     somelist += [3,4,5]
#     print somelist

# funa(lista[:])
# print(lista)



#***********************************************************************************************


#***********************************************************************************************
# #saving and retrieving numpy arrays to txt
# c1 = np.arange(0,8,1)
# c2 = np.arange(5,13,1)
# c3 = np.arange(20,28,1)
# # c4 = np.arange(20,28,1)

# # a = np.arange(27)
# # a = a.reshape(-1,3)
# # print('a', a.shape, a)


# # #set the directory name you want to give
# # dir_path = os.path.join('./data', 'metadata')

# # #if that directory doesnt already exists create one
# # if not os.path.exists(os.path.dirname(dir_path)):
# #     try:
# #         folder = os.mkdir(dir_path)
# #     except OSError as exc: # Guard against race condition
# #         if exc.errno != errno.EEXIST:
# #             raise
# # else:
# #     folder = dir_path
# #     print(folder)

# # #Create a file name in the created directory
# # extension = '.txt'
# # filename = os.path.join(folder, '0002937'+ extension)

# # #Write to the file
# # np.savetxt(filename, (c1,c2,c3,c4))
# np.save('./array_saved.npy', (c1,c2,c3))

# #load the file and read the values
# #loaded_values = np.loadtxt(filename)
# loaded_values = np.load('./array_saved.npy')

# print('loaded value shape, length values:', loaded_values.shape, len(loaded_values), loaded_values)

# #print individual elements of the file
# for i in range(len(loaded_values)):
#     print('loaded value %s: %s' %(i, loaded_values[i]) )

#***********************************************************************************************

# #***********************************************************************************************
# #print files in a directory

# directory = './data/test_data/'
# for filename in os.listdir(directory):
#     if filename.endswith(".bin") or filename.endswith(".py"):
#         #just the file name with extension
#         print(filename)

#         #complete path
#         print(os.path.join(directory, filename))
        
#         #just the name no extension
#         #print( ntpath.basename(os.path.join(directory, filename)).split('.')[0])
#         #continue
#     else:
#         continue

# #***********************************************************************************************

# #***********************************************************************************************
# #numpy voxel tests
#x = np.array([[0.2345, 0.2347, 0.2348 ], 
#               [0.3345, 0.3347, 0.3348 ],
#               [0.4345, 0.4347, 0.4348] ])
#
#min_x = x.min(0) #here 0 is the axis    
#print('minx', min_x)
#max_x = x.max(0)
#print('max', max_x)
#
#margin = max(max_x - min_x) - (max_x - min_x)
#xyzmin = min_x - margin / 2
#xyzmax = max_x + margin / 2
#
#print('margin', margin)
#print('xyzmin', xyzmin)
#print('xyzmax', xyzmax)
#
#n_x, n_y, n_z =  5,5,5
#x_y_z = [n_x, n_y, n_z]
#segments = []
#for i in range(3):
#    # note the +1 in num
#    s, step = np.linspace(xyzmin[i], xyzmax[i], num=(x_y_z[i] + 1), retstep=True)
#    segments.append(s)
#    
#def cartesian(arrays, out=None):
#    """Generate a cartesian product of input arrays.
#
#    Parameters
#    ----------
#    arrays : list of array-like
#        1-D arrays to form the cartesian product of.
#    out : ndarray
#        Array to place the cartesian product in.
#
#    Returns
#    -------
#    out : ndarray
#        2-D array of shape (M, len(arrays)) containing cartesian products
#        formed of input arrays.
#
#    Examples
#    --------
#    >>> cartesian(([1, 2, 3], [4, 5], [6, 7]))
#    array([[1, 4, 6],
#           [1, 4, 7],
#           [1, 5, 6],
#           [1, 5, 7],
#           [2, 4, 6],
#           [2, 4, 7],
#           [2, 5, 6],
#           [2, 5, 7],
#           [3, 4, 6],
#           [3, 4, 7],
#           [3, 5, 6],
#           [3, 5, 7]])
#
#    """
#    arrays = [np.asarray(x) for x in arrays]
#    shape = (len(x) for x in arrays)
#    dtype = arrays[0].dtype
#
#    ix = np.indices(shape)
#    ix = ix.reshape(len(arrays), -1).T
#
#    if out is None:
#        out = np.empty_like(ix, dtype=dtype)
#
#    for n, arr in enumerate(arrays):
#        out[:, n] = arrays[n][ix[:, n]]
#
#    return out
#    
#    ##np.digitize(value, (bins[1:] + bins[:-1])/2.0)
#
## bins = np.array([0.0, 1.0, 2.5, 4.0, 10.0])
## inds = np.digitize(x, bins)
## print('inds', inds)
#
## m = np.searchsorted(x,0.2440)
## print('m', m)

#*************************************************************************************************
