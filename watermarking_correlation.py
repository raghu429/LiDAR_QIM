#!/usr/bin/env python
import sys
import numpy as np
import tensorflow as tf
from helper_functions import *
from matplotlib import pyplot
import glob
import os
import pcl


# '''
# 1. Divide the frame into zones  (initially based on the x distance) and identify the zone
# 2. Add different random gaussian noise to different zones
# 3. Run through the model and check if the prediction is intact
# 4. Decode and extract the watermark 
# 5. check if this method detects  distortion in a particular zone

# encoding and decoding mechanism
# Ecnode
#     w = watermark or gaussian noise in our case
#     c = point cloud 
#     wc = watermarked point cloud
#     wc = w + c
# Decode
#     wc = watermarked point cloud
#     w = reference watermark ( use same seed as the encoder when generating the gaussian noise)
#     lc = linear correlation between watermark 'w' and the 'wc' is > threshold (0,7) then watermark detected
#     lc (w, wc) at a given zone  = sum((wi.wci))/length(w) 
#     Here we could do this correlation at each zone and check if the water marking is getting detected in each zone
#     with mormal point cloud and altered (additional car added)

# '''

def add_gaussian_noise(seed, data, mean_noise, std_dev_noise):
    #initate the random generator with seed
    np.random.seed(seed)
    noise = np.random.normal(mean_noise, std_dev_noise, data.shape)
    #print('noise shape', noise.shape)
    #print('noise data:', noise[:3, :])
    noisy_data = data + noise
    return noisy_data

def measure_distortion(pointCloud_base, pointCloud_modified):
    #print('base shape:', pointCloud_base.shape)
    #print('modified shape:', pointCloud_modified.shape)
    if(pointCloud_base.shape != pointCloud_modified.shape):
        print('point cloud shapes do not match')
    
    rmse = 0.0
    for i in range(0, pointCloud_base.shape[0]):
        rmse = rmse + eucledian_distance(pointCloud_base[i],pointCloud_modified[i])
    rmse = np.sqrt(rmse/pointCloud_base.shape[0]).astype(np.float32)
    return rmse

    #if(np.isnan())
    #if there are any nans in the data skip them.. but we could do this pre-processing before hand

def eucledian_distance(pointa, pointb):
    return(np.sqrt( (pointa[0]-pointb[0])**2 + (pointa[1]-pointb[1])**2 +(pointa[2]-pointb[2])**2))

def linear_correlation(a,b):
    #compare the sizes
    a_size = len(a)
    b_size = len(b)

    print('a size', a_size)
    #print('a:', a)
    print('b size', b_size)
    #print('b:', b)

    if(a_size != b_size):
        raise Exception('inputs have different size')
    #initialize the correlation
    lin_corr = 0
    for i in range(a_size):
        lin_corr += a[i]*float(b[i])
    return(lin_corr/(a_size))

''''
How to convert a point cloud to a voxel
1. first set the boundaries of the point cloud and filter unwanted points
2. then say if you want the resolution of 0.1 substract the min value in x,y,z from the point cloud and divide with the resolution
3. now make a voxel with given resolution of the length x,y,and z/resolution
4. light up the points in the voxel 

'''
def convert_to_voxel(pc,resolution=0.001):
    print('pc shape', pc.shape)
    print('pc x min:', np.min(pc[:,0]))

    x0 = np.min(pc[:,0]).astype(np.float32)
    y0 = np.min(pc[:,1]).astype(np.float32)
    z0 = np.min(pc[:,2]).astype(np.float32)

    x1 = np.max(pc[:,0]).astype(np.float32)
    y1 = np.max(pc[:,1]).astype(np.float32)
    z1 = np.max(pc[:,2])

    
    print('x0,y0,z0:', x0,y0,z0)
    print('x1,y1,z1:', x1,y1,z1)

    pc =((pc - np.array([x0, y0, z0 ])) / resolution).astype(np.int32)

    print('pc shape after quantization:', pc.shape)
    print('pc after quantization first 100 elements', pc[:100, :])
    #pc_logical =[(((pc - np.array([x0, y0, z0])) / resolution).astype(np.float32) > 0.05)]

    # make a voxel with similar shape .. by dividing the x,y,z length with the resolution
    voxel = np.zeros((int((x1 - x0) / resolution)+1, int((y1 - y0) / resolution)+1, int(round((z1-z0) / resolution)+1)))
    #print('insdie the drawvoxel function')
    print('voxel shape defined', voxel.shape)
    print ('voxel defined first hundred ', voxel[:-100,:2,:])


    #here we light up the points in the voxel that correspond to the point cloud
    voxel[pc[:, 0], pc[:, 1], pc[:, 2]] = 1
    
    print('voxel of pc elements shape', voxel.shape)
    #print('voxel of pc elements first 100', voxel[:-100, :2, :])
    print('voxel of pc elements first 100', voxel[57, 113,  55])
    print('voxel of pc elements first 100', voxel[56, 113,  55])
    print('voxel of pc elements first 100', voxel[55, 113,  55])
    
    return voxel


if __name__ == '__main__':
    
    #load the point cloud
    dataformat  =  "bin";
    data_path = "./data/002937.bin"

    if dataformat == "bin":
        pc_in = load_pc_from_bin(data_path)
    
    #do the voxeliazation
    #voxel_pc = convert_to_voxel(pc_in, 0.001)

    max_x0 = np.max(pc_in[:,0]).astype(np.float32)
    count = 0
    print ('pc_in size', pc_in.shape[0])

    first_third_cont = 0
    second_third_cont = 0
    third_third_cont = 0

    for i in range (0, pc_in.shape[0]):
        #find the max of x and then divide it into three parts
        x_val = pc_in[i,0]
        if (x_val <= (1/3)*max_x0):
            np.random.seed(100)
            #noise = np.random.normal(0, 0.001, int((1/3)*xmax))
            #pc_in[:, 2] = pc_in[:, 2] + noise
            first_third_cont += 1
        if(x_val> (1/3)*max_x0 and x_val <= (2/3)*max_x0):
            #np.random.seed(200)
            #noise = np.random.normal(0, 0.002, int((1/3)*max_x0))
            #pc_in[:, 2] = pc_in[:, 2] + noise
            second_third_cont += 1
        if(x_val > (2/3)*max_x0 and x_val <= max_x0):
            #np.random.seed(200)
            #noise = np.random.normal(0, 0.003, int((1/3)*max_x0))
            #pc_in[:, 2] = pc_in[:, 2] + noise
            third_third_cont += 1

    print('first third length', first_third_cont)
    print('second third length', second_third_cont)
    print('thirdthird length', third_third_cont)
    
    np.random.seed(100)
    noise_first = np.random.normal(0, 0.3, first_third_cont)

    np.random.seed(200)
    noise_second = np.random.normal(1, 0.3, second_third_cont)

    np.random.seed(300)
    noise_third = np.random.normal(2, 0.3, third_third_cont)

    out_pc = pc_in
    out_pc[:first_third_cont, 2] = pc_in[:first_third_cont, 2] + noise_first
    out_pc[first_third_cont:third_third_cont, 2] = pc_in[first_third_cont:third_third_cont, 2] + np.random.normal(2, 0.3, third_third_cont-first_third_cont)


    #calculate the distortion coefficient (optional)
    #print('rmse:', measure_distortion(pc_in[:first_third_cont, 0] , out_pc[:first_third_cont, 0]))
    print('rmse:', measure_distortion(pc_in , out_pc))
    

    #correlate this with the noise and see if you get the peak
    np.random.seed(100)
    noise_first_regenerated = np.random.normal(1, 0.3, first_third_cont)

    noise_third_regenerated = np.random.normal(0, 0.3, third_third_cont-first_third_cont)

    #do the correlation
    lcs_myfunc = linear_correlation(noise_first,out_pc[:first_third_cont, 2])
    print('lcs_myfunc', lcs_myfunc)

    lcs = np.correlate(noise_third_regenerated,out_pc[first_third_cont:third_third_cont, 2])

    print('lcs', lcs/len(noise_first_regenerated))

    #correlation_range_max = max_x0;
    #print('the max value of the x', correlation_range_max)
    #print('the max value of point cloud condidered', (1/3.0)*max_x0)

    # #plot the correlation
    # # bins = np.linspace(-correlation_range_max, correlation_range_max, 500)
    # # pyplot.hist(lcs, bins, alpha=0.5, label='lc')
    # # pyplot.show()
    
    # #uncomment the following lines to understand the pyplot
    # x = [np.random.normal(0,1) for _ in range(400)]
    # print('x range', np.ptp(x))
    # print('x length & x', len(x))
    # print(' x min', np.min(x))
    # print(' x max', np.max(x))


    # y = [np.random.normal(0,5) for _ in range(400)]
    # print('y lenght & y', len(y))
    # print('y range', np.ptp(y))
    # print(' y min', np.min(y))
    # print(' y max', np.max(y))

    # bins = np.linspace(-10, 10, 1000)

    # pyplot.hist(x, bins, alpha=1, label='x')
    # pyplot.hist(y, bins, alpha=1, label='y')
    # pyplot.legend(loc='upper right')
    
    # # pyplot.show()

    #  #do the correlation
    # lcs1 = linear_correlation(x,y)
    # print('lcs1', lcs1)
    
    # pears = pearsonr(x,y)
    # print('pears', pears)

    # lcs = np.correlate(x,y)

    # print('lcs', lcs/len(x))
    # bins = np.linspace(-1, 1, 100)
    # pyplot.hist(lcs1, bins, alpha=1, label='lc')

    # pyplot.show()
    #save the voxelized point cloud for visualization
    #for each data data item divide the point cloud into three sections based on the x distance and add a guassian noise parameter based on the different seed
    #pcl.save(voxel_pc, "bunny_voxel.pcd")
    
    # while not rospy.is_shutdown():

    #     pub1 = rospy.Publisher("/points_raw", PointCloud2, queue_size=1000000)
    #     rospy.init_node("pc2_publisher")
    #     header = std_msgs.msg.Header()
    #     header.stamp = rospy.Time.now()
    #     header.frame_id = "velodyne"
    #     points = pc2.create_cloud_xyz32(header, voxel_pc[:, :3].astype(np.float32))

    # rospy.spin()

 





