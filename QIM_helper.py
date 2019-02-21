#!/usr/bin/env python
import numpy as np
import time
import math
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



pc_test = np.array([[52.249, 12.482,  2.026],
       [51.769, 12.453,  2.01 ],
       [51.275, 12.505,  1.994],
       [50.792, 12.556,  1.978],
       [50.31 , 12.604,  1.963],
       [49.84 , 12.653,  1.948],
       [49.363, 12.697,  1.933]]).astype(np.float64)
    #    [48.903, 12.742,  1.918],
    #    [48.459, 12.708,  1.903],
    #    [48.042, 12.759,  1.89 ],
    #    [47.593, 12.8  ,  1.876],
    #    [47.153, 12.841,  1.862],
    #    [46.735, 12.884,  1.848],
    #    [46.288, 12.918,  1.834],
    #    [46.201, 13.05 ,  1.833],
    #    [46.235, 13.138,  1.834],
    #    [45.114, 12.972,  1.796],
    #    [44.86 , 13.051,  1.789],
    #    [44.632, 13.137,  1.782],
    #    [44.38 , 13.215,  1.775],
    #    [44.197, 13.311,  1.77 ],
    #    [44.011, 13.331,  1.764],
    #    [43.851, 13.433,  1.76 ],
    #    [43.835, 13.579,  1.761],
    #    [43.815, 13.723,  1.762],
    #    [43.823, 13.877,  1.764],
    #    [43.844, 14.036,  1.766],
    #    [43.855, 14.191,  1.768],
    #    [43.884, 14.277,  1.77 ],
    #    [43.894, 14.433,  1.772],
    #    [43.922, 14.595,  1.774],
    #    [43.943, 14.755,  1.777],
    #    [43.977, 14.921,  1.779],
    #    [44.023, 15.091,  1.783],
    #    [56.316, 19.511,  2.224],
    #    [55.898, 19.563,  2.212],
    #    [55.498, 19.52 ,  2.198],
    #    [55.078, 19.567,  2.186],
    #    [54.672, 19.616,  2.173],
    #    [54.263, 19.662,  2.161]]).astype(float)


# resolution = 0.025
# resolution = 0.05

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


x = (0,1.4)
y = (0,1.2)
z = (0,1.5)

resolution_delta = 0.05
resolution_halfdelta = resolution_delta/2.0

def Hausdorff_dist_simple(vol_a,vol_b):
    dist_lst = []
    print('vola length', len(vol_a))
    print('volb length', len(vol_b)
    )
    for idx in range(len(vol_a)):
        dist_min = 1000.0
        print(idx)        
        for idx2 in range(len(vol_b)):
            # print(idx2)
            dist= np.linalg.norm(vol_a[idx]-vol_b[idx2])
            if dist_min > dist:
                dist_min = dist
        dist_lst.append(dist_min)
    return np.max(dist_lst)

def bbox(array, point, radius):
    a = array[np.where(np.logical_and(array[:, 0] >= point[0] - radius, array[:, 0] <= point[0] + radius))]
    b = a[np.where(np.logical_and(a[:, 1] >= point[1] - radius, a[:, 1] <= point[1] + radius))]
    c = b[np.where(np.logical_and(b[:, 2] >= point[2] - radius, b[:, 2] <= point[2] + radius))]
    return c

def Hausdorff_dist(surface_a, surface_b):

    # Taking two arrays as input file, the function is searching for the Hausdorff distane of "surface_a" to "surface_b"
    dists = []

    l = len(surface_a)

    start = time.time()
    for i in range(l):
        # print(i)    
        # walking through all the points of surface_a
        dist_min = 1000.0
        radius = 0
        b_mod = np.empty(shape=(0, 0, 0))

        # increasing the cube size around the point until the cube contains at least 1 point
        while b_mod.shape[0] == 0:
            b_mod = bbox(surface_b, surface_a[i], radius)
            radius += 1

        # to avoid getting false result (point is close to the edge, but along an axis another one is closer),
        # increasing the size of the cube
        b_mod = bbox(surface_b, surface_a[i], radius * math.sqrt(3))

        for j in range(len(b_mod)):
            # walking through the small number of points to find the minimum distance
            dist = np.linalg.norm(surface_a[i] - b_mod[j])
            if dist_min > dist:
                dist_min = dist
        
        dists.append(dist_min)
    
    end = time.time()
    # print('time took:', end-start)
    return np.max(dists)


def get_boxcorners(places, rotates, size):
    """Create 8 corners of bounding box from bottom center."""
    corners = []
    for place, rotate, sz in zip(places, rotates, size):
        x, y, z = place
        h, w, l = sz
        if l > 10:
            continue

        corner = np.array([
            [x - l / 2., y - w / 2., z],
            [x + l / 2., y - w / 2., z],
            [x - l / 2., y + w / 2., z],
            [x - l / 2., y - w / 2., z + h],
            [x - l / 2., y + w / 2., z + h],
            [x + l / 2., y + w / 2., z],
            [x + l / 2., y - w / 2., z + h],
            [x + l / 2., y + w / 2., z + h],
        ])
        # corner = np.array([
        #     [x - l / 2., y - w / 2., z], #bottom surface bottom right (0)
        #     # [x + l / 2., y - w / 2., z], #bottom surface top right (1)
        #     [x - l / 2., y + w / 2., z], #bottom surface bottom left (2)
        #     [x - l / 2., y - w / 2., z + h],#bottom right (3)
        #     [x - l / 2., y + w / 2., z + h],#bottom left (4)
        #     # [x + l / 2., y + w / 2., z],#bottom surface top left (5)
        #     # [x + l / 2., y - w / 2., z + h],# top right (6)
        #     # [x + l / 2., y + w / 2., z + h],# top left (7)
        #     ])

        corner -= np.array([x, y, z])

        rotate_matrix = np.array([
            [np.cos(rotate), -np.sin(rotate), 0],
            [np.sin(rotate), np.cos(rotate), 0],
            [0, 0, 1]
        ])

        a = np.dot(corner, rotate_matrix.transpose())
        a += np.array([x, y, z])
        corners.append(a)
    return np.array(corners).astype(np.float32)



    def get_boxcorners_front(places, rotates, size):
        """Create 8 corners of bounding box from bottom center."""
        corners = []
        for place, rotate, sz in zip(places, rotates, size):
            x, y, z = place
            h, w, l = sz
            if l > 10:
                continue

            # corner = np.array([
            #     [x - l / 2., y - w / 2., z],
            #     [x + l / 2., y - w / 2., z],
            #     [x - l / 2., y + w / 2., z],
            #     [x - l / 2., y - w / 2., z + h],
            #     [x - l / 2., y + w / 2., z + h],
            #     [x + l / 2., y + w / 2., z],
            #     [x + l / 2., y - w / 2., z + h],
            #     [x + l / 2., y + w / 2., z + h],
            # ])
            corner = np.array([
                [x - l / 2., y - w / 2., z], #bottom surface top right (0)
                # [x + l / 2., y - w / 2., z], #bottom surface bottom right (1)
                [x - l / 2., y + w / 2., z], #bottom surface top left (2)
                [x - l / 2., y - w / 2., z + h],#top surface top right (3)
                [x - l / 2., y + w / 2., z + h],#top surface top left (4)
                # [x + l / 2., y + w / 2., z],#bottom surface bottom left (5)
                # [x + l / 2., y - w / 2., z + h],# top surface bottom right (6)
                # [x + l / 2., y + w / 2., z + h],# top surface bottom left (7)
                ])

        #          corner = np.array([
        # [x, y, z], #bottom surface bottom left (0) 
        # [x, yy, z], #bottom surface right (1)
        # [xx, y, z], #bottom surface top left (2)(2)
        # [xx, yy, z],#bottom surface top right (3)(0)
        # [x, y, zz], #top surface left (4)
        # [x, yy, zz], #top surface right (5)
        # [xx, y, zz], #top surface top left (6)(4)
        # [xx, yy, zz],#top surface top right (7)(3)
        # ])

            corner -= np.array([x, y, z])

            rotate_matrix = np.array([
                [np.cos(rotate), -np.sin(rotate), 0],
                [np.sin(rotate), np.cos(rotate), 0],
                [0, 0, 1]
            ])

            a = np.dot(corner, rotate_matrix.transpose())
            a += np.array([x, y, z])
            corners.append(a)
        return np.array(corners).astype(np.float32)

def read_label_from_txt(label_path):
    """Read label from txt file."""
    text = np.fromfile(label_path)
    bounding_box = []
    with open(label_path, "r") as f:
        labels = f.read().split("\n")
        for label in labels:
            if not label:
                continue
            label = label.split(" ")
            if (label[0] == "DontCare"):
                continue

            if label[0] == ("Car" or "Van"): #  or "Truck"
                bounding_box.append(label[8:15])

    if bounding_box:
        data = np.array(bounding_box, dtype=np.float32)
        return data[:, 3:6], data[:, :3], data[:, 6]
    else:
        return None, None, None

def read_calib_file(calib_path):
    """Read a calibration file."""
    data = {}
    with open(calib_path, 'r') as f:
        for line in f.readlines():
            if not line or line == "\n":
                continue
            key, value = line.split(':', 1)
            try:
                data[key] = np.array([float(x) for x in value.split()])
            except ValueError:
                pass
    return data

def read_labels(label_path, label_type, calib_path=None, is_velo_cam=False, proj_velo=None):
    """Read labels from xml or txt file.
    Original Label value is shifted about 0.27m from object center.
    So need to revise the position of objects.
    """
    if label_type == "txt": #TODO
        places, size, rotates = read_label_from_txt(label_path)
        if places is None:
            return None, None, None
        rotates = np.pi / 2 - rotates
        dummy = np.zeros_like(places)
        dummy = places.copy()
        if calib_path:
            places = np.dot(dummy, proj_velo.transpose())[:, :3]
        else:
            places = dummy
        if is_velo_cam:
            places[:, 0] += 0.27

    elif label_type == "xml":
        bounding_boxes, size = read_label_from_xml(label_path)
        places = bounding_boxes[30]["place"]
        rotates = bounding_boxes[30]["rotate"][:, 2]
        size = bounding_boxes[30]["size"]

    return places, rotates, size

def proj_to_velo(calib_data):
    """Projection matrix to 3D axis for 3D Label"""
    rect = calib_data["R0_rect"].reshape(3, 3)
    #to transform a  point from Lidar framce to camera frame
    #reshape the flat line with 12 elements to 3X4 matrix
    velo_to_cam = calib_data["Tr_velo_to_cam"].reshape(3, 4)
#print('velo2cam', velo_to_cam)
    inv_rect = np.linalg.inv(rect)
    #select all rows and only first three columns
#print('velo_to_cam[:, :3]', velo_to_cam[:, :3])
    #select all rows and only first three columns
    inv_velo_to_cam = np.linalg.pinv(velo_to_cam[:, :3])
    return np.dot(inv_velo_to_cam, inv_rect)

def visualize_groundtruth(calib_path, label_path):
    #visualize the gorund truth
    if calib_path:
        calib = read_calib_file(calib_path)
        proj_velo = proj_to_velo(calib)[:, :3]
    #print('proj_velo', proj_velo)
    if label_path:
        places, rotates, size = read_labels(label_path, label_type="txt", calib_path=calib_path, is_velo_cam=False, proj_velo=proj_velo)
    #print('places, rotates, size', places, rotates, size)
    
    corners_gt = get_boxcorners(places, rotates, size)
    # print('gt corners shape', corners_gt.shape )
    # print('gt locations:', places)
    return corners_gt

def get_boundingboxcorners(cluster_points):
    #print('point min', point_min)    
    #print('point max', point_max)
    #corner = 0
    cluster_points = cluster_points.reshape(-1, 3)
    x_min = cluster_points[:,0].min()
    y_min = cluster_points[:,1].min()
    z_min = cluster_points[:,2].min()

    x_max = cluster_points[:,0].max()
    y_max = cluster_points[:,1].max()
    z_max = cluster_points[:,2].max()

    # print('cluter point list min', x_min, y_min, z_min )
    # print('cluter point list min', x_max, y_max, z_max )
    if( (x_min != 0) & (x_max !=0 )):  #this condition has been added t remove centeroids at (0,0,0) appeared when the clusters were removed
        cluster_corner = get_boundingboxcorners_local([x_min, y_min, z_min], [x_max, y_max, z_max])
        #print('corner shape & value', corner.shape, corner)
        # corners = np.append(corners, [corner])
    return cluster_corner


def get_boundingboxcorners_local(point_min, point_max):

    x, y, z = point_min
    # print('x,y,z, min', x,y,z)
    xx, yy, zz = point_max
    # print('x,y,z, max', xx,yy,zz)

    #in the following sequence if we want to get the four corners of bottom surface (top view)
    # we need to extract the following rows [0,1, 2, 5]
    #for front view projection we need [0, 2, 3, 4]
    corner = np.array([
        [x, y, z], #bottom surface left (0)
        [x, yy, z], #bottom surface right (1)
        [xx, y, z], #bottom surface top left (2)
        [xx, yy, z],#bottom surface top right (3)
        [x, y, zz], #top surface left (4)
        [x, yy, zz], #top surface right (5)
        [xx, y, zz], #top surface top left (6)
        [xx, yy, zz],#top surface top right (7)
        ])

    return corner

def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    print('box A', boxA)
    print('box B', boxB)
    xA = max(boxA[0,0], boxB[0,0])
    yA = max(boxA[0,1], boxB[0,1])
    xB = min(boxA[1,0], boxB[1,0])
    yB = min(boxA[1,1], boxB[1,1])

    # compute the area of intersection rectangle
    interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))

    print('interA', interArea)
    if interArea == 0:
        return 0
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = abs((boxA[1,0] - boxA[0,0]) * (boxA[1,1] - boxA[0,1]))
    boxBArea = abs((boxB[1,0] - boxB[0,0]) * (boxB[1,1] - boxB[0,1]))

    print('boxAArea', boxAArea)
    print('boxBArea', boxBArea)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou


def Get_tamperindices_blockbased(decoded_code, num_bits):
    
    if(num_bits == 3):
        template_code = np.array([[0,0,0],
                            [0,0,1],
                            [0,1,0],
                            [0,1,1],
                            [1,0,0],
                            [1,0,1],
                            [1,1,0],
                            [1,1,1]])
    elif(num_bits == 2):
        template_code = np.array([[0,0],
                            [0,1],
                            [1,0],
                            [1,1]])
    elif(num_bits == 1):
        template_code = np.array([[0],[1]])
    else:
        print("WRONG number of bits inputted")
    

    suspect_indices = []

    # print('template code size', template_code.shape[0])
    
    row_count = 0
    block_move_size = template_code.shape[0]

    row_count_increment = block_move_size
    # block_traverse_size = decoded_code.shape[0] - row_count_increment
    

    while (row_count < decoded_code.shape[0]):
        # print('block count, row_count_increment', row_count, row_count_increment)

        block_move_length = (row_count + block_move_size)

        if(block_move_length - decoded_code.shape[0] > block_move_size):
            block_move_size = 1
            block_move_length = (row_count + block_move_size)
            
        mod_block = decoded_code[row_count:block_move_length, :]
        
        if(np.array_equal(mod_block, template_code) == False):
            row_count_increment = 1
            #mark the indices
            suspect_indices.append(row_count)
            # print('suspect index list:', suspect_indices)
        else:
            row_count_increment = block_move_size

        #advance the index
        row_count  += row_count_increment
    
    if(len(suspect_indices) == 0):
        print('PC intact')
    else:
        print('PC compromised')
        suspect_indices = np.array(suspect_indices)

    return suspect_indices

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
    return (np.array(cluster_min_max).reshape(-1,2) )

        
def get_tamperedindices_sequential_threebits(decoded_codebook):
    
    encode_cb_0 = np.array([0,0,0])
    encode_cb_1 = np.array([0,0,1])
    encode_cb_2 = np.array([0,1,0])
    encode_cb_3 = np.array([0,1,1])

    encode_cb_4 = np.array([1,0,0])
    encode_cb_5 = np.array([1,0,1])
    encode_cb_6 = np.array([1,1,0])
    encode_cb_7 = np.array([1,1,1])
    
    cloud_to_compare = []
    error_counter = 0

    suspect_indices = []
    for index in range(len(decoded_codebook)):
        
        # index = indices[i]
        cloud_to_compare = decoded_codebook[index]
        
        if(np.mod(index, 8) == 0): # 0,8, 16...
            # print('encode_0', encode_cb_0)
            if( (cloud_to_compare == encode_cb_0).all()):
                # print('**********condition exists [0,0,0]')
                pass
            else:
                suspect_indices.append(index)    
                changed_indices = np.where(cloud_to_compare != encode_cb_0)
                for i in range(len(changed_indices[0])):
                    # increment the error counter
                    error_counter += 1
        
        elif(np.mod(index, 8) == 1): # 1,9, ...
            # print('encodecb_1', encode_cb_1)
            if( (cloud_to_compare == encode_cb_1).all()):
                pass
                # print('**********condition exisits [0,0,1]')
            else:
                suspect_indices.append(index)
                changed_indices = np.where(cloud_to_compare != encode_cb_1)
                for i in range(len(changed_indices[0])):
                    # increment the error counter
                    error_counter += 1
        
        elif(np.mod(index, 8) == 2): # 2,10 ...
            # print('encodecb_2', encode_cb_2)
            if( (cloud_to_compare == encode_cb_2).all()):
                pass
                # print('**********condition exisits [0,1,0]')
            else:
                suspect_indices.append(index)
                changed_indices = np.where(cloud_to_compare != encode_cb_2)
                for i in range(len(changed_indices[0])):
                    # increment the error counter
                    error_counter += 1
        
        elif(np.mod(index, 8) == 3): # 3,11 ...
            # print('encodecb_3', encode_cb_3)
            if( (cloud_to_compare == encode_cb_3).all()):
                pass
                # print('**********condition exisits [0,1,1]')
            else:
                suspect_indices.append(index)
                changed_indices = np.where(cloud_to_compare != encode_cb_3)
                for i in range(len(changed_indices[0])):
                    # increment the error counter
                    error_counter += 1
        
        elif(np.mod(index, 8) == 4): # 4,12 ...
            # print('encodecb_4', encode_cb_4)
            if( (cloud_to_compare == encode_cb_4).all()):
                pass
                # print('**********condition exisits [1,0,0]')
            else:
                suspect_indices.append(index)
                changed_indices = np.where(cloud_to_compare != encode_cb_4)
                for i in range(len(changed_indices[0])):
                    # increment the error counter
                    error_counter += 1
        
        elif(np.mod(index, 8) == 5): # 5,13 ...
            # print('encodecb_5', encode_cb_5)
            if( (cloud_to_compare == encode_cb_5).all()):
                pass
                # print('**********condition exisits [1,0,1]')
            else:
                suspect_indices.append(index)
                changed_indices = np.where(cloud_to_compare != encode_cb_5)
                for i in range(len(changed_indices[0])):
                    # increment the error counter
                    error_counter += 1
        
        elif(np.mod(index, 8) == 6): # 6,14 ...
            # print('encodecb_6', encode_cb_6)
            if( (cloud_to_compare == encode_cb_6).all()):
                pass
                # print('**********condition exisits [1,1,0]')
            else:
                suspect_indices.append(index)
                changed_indices = np.where(cloud_to_compare != encode_cb_6)
                for i in range(len(changed_indices[0])):
                    # increment the error counter
                    error_counter += 1
        
        elif(np.mod(index, 8) == 7): # 7,15 ...
            # print('encodecb_7', encode_cb_7)
            if( (cloud_to_compare == encode_cb_7).all()):
                pass
                # print('**********condition exisits [1,1,1]')
            else:
                suspect_indices.append(index)
                changed_indices = np.where(cloud_to_compare != encode_cb_7)
                for i in range(len(changed_indices[0])):
                    # increment the error counter
                    error_counter += 1
        
    # if(len(suspect_indices) == 0):
    #     print('PC intact')
    # else:
    #     print('PC compromised')
    suspect_indices = np.array(suspect_indices)
    lensuspect = len(suspect_indices) 
    
    if(lensuspect == 0):
        lensuspect = 0.1
    error_rate =  error_counter/(3.0*len(decoded_codebook))

    return suspect_indices, error_rate 
        
        
        
    #     print('suspect indices final', suspect_indices)
    #     tampered_pc_indices = get_tamperedpc_indices(suspect_indices, 2)
    #     # print('tampered cluster indices', tampered_pc_indices)
    
    # # return tampered_pc_indices

def get_BER_threebits(indices, decoded_codebook):
    
    encode_cb_0 = np.array([0,0,0])
    encode_cb_1 = np.array([0,0,1])
    encode_cb_2 = np.array([0,1,0])
    encode_cb_3 = np.array([0,1,1])

    encode_cb_4 = np.array([1,0,0])
    encode_cb_5 = np.array([1,0,1])
    encode_cb_6 = np.array([1,1,0])
    encode_cb_7 = np.array([1,1,1])
    
    cloud_to_compare = []
    error_counter = 0
    index = 0

    for i in range(len(indices)):
        
        index = indices[i]
        cloud_to_compare = decoded_codebook[index, :]
        
        if(np.mod(index, 8) == 0): # 0,8, 16...
            # print('encode_0', encode_cb_0)
            if( (cloud_to_compare == encode_cb_0).all()):
                # print('**********condition exists [0,0,0]')
                pass
            else:
                changed_indices = np.where(cloud_to_compare != encode_cb_0)
                for i in range(len(changed_indices[0])):
                    # increment the error counter
                    error_counter += 1
        
        elif(np.mod(index, 8) == 1): # 1,9, ...
            # print('encodecb_1', encode_cb_1)
            if( (cloud_to_compare == encode_cb_1).all()):
                pass
                # print('**********condition exisits [0,0,1]')
            else:
                changed_indices = np.where(cloud_to_compare != encode_cb_1)
                for i in range(len(changed_indices[0])):
                    # increment the error counter
                    error_counter += 1
        
        elif(np.mod(index, 8) == 2): # 2,10 ...
            # print('encodecb_2', encode_cb_2)
            if( (cloud_to_compare == encode_cb_2).all()):
                pass
                # print('**********condition exisits [0,1,0]')
            else:
                changed_indices = np.where(cloud_to_compare != encode_cb_2)
                for i in range(len(changed_indices[0])):
                    # increment the error counter
                    error_counter += 1
        
        elif(np.mod(index, 8) == 3): # 3,11 ...
            # print('encodecb_3', encode_cb_3)
            if( (cloud_to_compare == encode_cb_3).all()):
                pass
                # print('**********condition exisits [0,1,1]')
            else:
                changed_indices = np.where(cloud_to_compare != encode_cb_3)
                for i in range(len(changed_indices[0])):
                    # increment the error counter
                    error_counter += 1
        
        elif(np.mod(index, 8) == 4): # 4,12 ...
            # print('encodecb_4', encode_cb_4)
            if( (cloud_to_compare == encode_cb_4).all()):
                pass
                # print('**********condition exisits [1,0,0]')
            else:
                changed_indices = np.where(cloud_to_compare != encode_cb_4)
                for i in range(len(changed_indices[0])):
                    # increment the error counter
                    error_counter += 1
        
        elif(np.mod(index, 8) == 5): # 5,13 ...
            # print('encodecb_5', encode_cb_5)
            if( (cloud_to_compare == encode_cb_5).all()):
                pass
                # print('**********condition exisits [1,0,1]')
            else:
                changed_indices = np.where(cloud_to_compare != encode_cb_5)
                for i in range(len(changed_indices[0])):
                    # increment the error counter
                    error_counter += 1
        
        elif(np.mod(index, 8) == 6): # 6,14 ...
            # print('encodecb_6', encode_cb_6)
            if( (cloud_to_compare == encode_cb_6).all()):
                pass
                # print('**********condition exisits [1,1,0]')
            else:
                changed_indices = np.where(cloud_to_compare != encode_cb_6)
                for i in range(len(changed_indices[0])):
                    # increment the error counter
                    error_counter += 1
        
        elif(np.mod(index, 8) == 7): # 7,15 ...
            # print('encodecb_7', encode_cb_7)
            if( (cloud_to_compare == encode_cb_7).all()):
                pass
                # print('**********condition exisits [1,1,1]')
            else:
                changed_indices = np.where(cloud_to_compare != encode_cb_7)
                for i in range(len(changed_indices[0])):
                    # increment the error counter
                    error_counter += 1
        
        error_rate =  error_counter/(3.0*len(indices))

    return error_rate

def get_BER_twobits(indices, decoded_codebook):
    
    encode_cb_0 = np.array([0,0])
    encode_cb_1 = np.array([0,1])
    encode_cb_2 = np.array([1,0])
    encode_cb_3 = np.array([1,1])

    cloud_to_compare = []
    error_counter = 0
    index = 0

    for i in range(len(indices)):
        
        index = indices[i]
        cloud_to_compare = decoded_codebook[index, :]
        
        if(np.mod(index, 4) == 0): # 0,8, 16...
            # print('encode_0', encode_cb_0)
            if( (cloud_to_compare == encode_cb_0).all()):
                # print('**********condition exists [0,0,0]')
                pass
            else:
                changed_indices = np.where(cloud_to_compare != encode_cb_0)
                for i in range(len(changed_indices[0])):
                    # increment the error counter
                    error_counter += 1
        
        elif(np.mod(index, 4) == 1): # 1,9, ...
            # print('encodecb_1', encode_cb_1)
            if( (cloud_to_compare == encode_cb_1).all()):
                pass
                # print('**********condition exisits [0,0,1]')
            else:
                changed_indices = np.where(cloud_to_compare != encode_cb_1)
                for i in range(len(changed_indices[0])):
                    # increment the error counter
                    error_counter += 1
        
        elif(np.mod(index, 4) == 2): # 2,10 ...
            # print('encodecb_2', encode_cb_2)
            if( (cloud_to_compare == encode_cb_2).all()):
                pass
                # print('**********condition exisits [0,1,0]')
            else:
                changed_indices = np.where(cloud_to_compare != encode_cb_2)
                for i in range(len(changed_indices[0])):
                    # increment the error counter
                    error_counter += 1
        
        elif(np.mod(index, 4) == 3): # 3,11 ...
            # print('encodecb_3', encode_cb_3)
            if( (cloud_to_compare == encode_cb_3).all()):
                pass
                # print('**********condition exisits [0,1,1]')
            else:
                changed_indices = np.where(cloud_to_compare != encode_cb_3)
                for i in range(len(changed_indices[0])):
                    # increment the error counter
                    error_counter += 1
        
        error_rate =  error_counter/(2.0*len(indices))
    return error_rate

def get_BER_twobits(indices, decoded_codebook):
    
    encode_cb_0 = np.array([0])
    encode_cb_1 = np.array([1])
    

    cloud_to_compare = []
    error_counter = 0
    index = 0

    for i in range(len(indices)):
        
        index = indices[i]
        cloud_to_compare = decoded_codebook[index, :]
        
        if(np.mod(index, 2) == 0): # 0,8, 16...
            # print('encode_0', encode_cb_0)
            if( (cloud_to_compare == encode_cb_0).all()):
                # print('**********condition exists [0,0,0]')
                pass
            else:
                changed_indices = np.where(cloud_to_compare != encode_cb_0)
                for i in range(len(changed_indices[0])):
                    # increment the error counter
                    error_counter += 1
        
        elif(np.mod(index, 2) == 1): # 1,9, ...
            # print('encodecb_1', encode_cb_1)
            if( (cloud_to_compare == encode_cb_1).all()):
                pass
                # print('**********condition exisits [0,0,1]')
            else:
                changed_indices = np.where(cloud_to_compare != encode_cb_1)
                for i in range(len(changed_indices[0])):
                    # increment the error counter
                    error_counter += 1
        
        error_rate =  error_counter/(1.0*len(indices))
        
    return error_rate

   def qim_decode(pc, resolution):
    # We can check if each coordinate is even or odd.. while encoding 
    final_code = []
    cloud_row_read = []
    cloud_row_read = getQuantizedValues_from_pointCloud(pc, resolution, x,y,z)

    for j in range (cloud_row_read.shape[0]):
        row_odd_flags = []       
        row_odd_flags = get_oddflag(cloud_row_read[j])
        final_code.append(row_odd_flags)

    
    # print('******decoded quantized values', cloud_row_read.shape, cloud_row_read)
    
    return final_code, cloud_row_read

def getPointCloud_from_quantizedValues(pc_encoded, resolution_in, x_in, y_in, z_in):
    pc_value = (pc_encoded *resolution_in) + np.array([x_in[0],y_in[0],z_in[0]])
    return(pc_value)

def getQuantizedValues_from_pointCloud(pc, resolution_in, x_in,y_in,z_in):
    pc_quant_value = np.around((pc - np.array([x_in[0],y_in[0],z_in[0]])) / resolution_in).astype(np.int32)
    # pc_quant_value = ((pc - np.array([x_in[0],y_in[0],z_in[0]]))/resolution_in).astype(np.int32)
    return(pc_quant_value)


#checks if the quantized value is even or odd
def get_oddflag(pc_row):
    o_flag = [] 
    # print('pc input', pc_row)
    for i in range(len(pc_row)):
            #print('index: value:', (i,j), pc[j][i])
            #extract decimal number
            # number_dec = int(pc_row[i] / resolution)
            number_dec = int(pc_row[i])
            # print('num dec', number_dec)
            if (number_dec % 2 == 0): #even
                o_flag.append(0)
            else:
                o_flag.append(1) #odd 

    # final_code.append( get_decode_codebook(e_flag) )
    # print('even flag', e_flag)
    return o_flag
    

# def qim_quantize_restricted_threebits(pc_in):
    
#     encode_cb_0 = np.array([0,0,0])
#     encode_cb_1 = np.array([0,0,1])
#     encode_cb_2 = np.array([0,1,0])
#     encode_cb_3 = np.array([0,1,1])

#     encode_cb_4 = np.array([1,0,0])
#     encode_cb_5 = np.array([1,0,1])
#     encode_cb_6 = np.array([1,1,0])
#     encode_cb_7 = np.array([1,1,1])
    
#     pc_row_count = 0

#     quant_encoded = []
#     quant_encoded_halfdelta = []
#     cbook = []
#     # cbook_decoded = []
#     # pc_values_decoded = []
    
#     while pc_row_count < len(pc_in):
#         cloud_to_compare = []
#         cloud_row = []
#         cloud_row_halfdelta = []

#         pc = pc_in[pc_row_count, :]
#         # cloud_row_clean = ((pc - np.array([x[0],y[0],z[0]])) / resolution).astype(np.int32)
#         quantized_values_row = getQuantizedValues_from_pointCloud(pc, resolution_delta, x,y,z)
#         # print('clean', cloud_row_clean)
#         cloud_row =  quantized_values_row
#         # print('input cloud-Delta', cloud_row)
#         cloud_row_halfdelta =  2*cloud_row
#         # print('input cloud- Half Delta', cloud_row_halfdelta)

#         cloud_to_compare = np.array(get_oddflag(cloud_row_halfdelta))
#         # print('input oddflag', cloud_to_compare)

#         #reset code book
#         code_book = [0,0,0]
#         # print('cloud compare', cloud_to_compare)

#         if(np.mod(pc_row_count, 8) == 0): # 0,8, 16...
#             # print('encode_0', encode_cb_0)
#             if( (cloud_to_compare == encode_cb_0).all()):
#                 # print('**********condition exists [0,0,0]')
#                 pass
#             else:
#                 changed_indices = np.where(cloud_to_compare != encode_cb_0)
#                 # print('0 change indices & length', changed_indices, len(changed_indices[0]))
#                 # print('\n')
#                 for i in range(len(changed_indices[0])):
#                     if(encode_cb_0[changed_indices[0][i]] == 1):
#                         cloud_row[changed_indices[0][i]] = cloud_row[changed_indices[0][i]] + 1
#                         cloud_row_halfdelta[changed_indices[0][i]] = cloud_row_halfdelta[changed_indices[0][i]] + 1
#                     else:
#                         cloud_row[changed_indices[0][i]] = cloud_row[changed_indices[0][i]] - 1
#                         cloud_row_halfdelta[changed_indices[0][i]] = cloud_row_halfdelta[changed_indices[0][i]] - 1

#                 # print('condition [000]', (count), cloud_row)
#             code_book = encode_cb_0 #[0,0,0]   
        
#         elif(np.mod(pc_row_count, 8) == 1): # 1,9, ...
#             # print('encodecb_1', encode_cb_1)
#             if( (cloud_to_compare == encode_cb_1).all()):
#                 pass
#                 # print('**********condition exisits [0,0,1]')
#             else:
#                 changed_indices = np.where(cloud_to_compare != encode_cb_1)
#                 # print('1 changed indices**', changed_indices)
#                 for i in range(len(changed_indices[0])):
#                     if(encode_cb_1[changed_indices[0][i]] == 1):
#                         cloud_row[changed_indices[0][i]] = cloud_row[changed_indices[0][i]] + 1
#                         cloud_row_halfdelta[changed_indices[0][i]] = cloud_row_halfdelta[changed_indices[0][i]] + 1
#                     else:
#                         cloud_row[changed_indices[0][i]] = cloud_row[changed_indices[0][i]] - 1
#                         cloud_row_halfdelta[changed_indices[0][i]] = cloud_row_halfdelta[changed_indices[0][i]] - 1

#                 # print('condition [001]', (count), cloud_row)
#             code_book = encode_cb_1 #[0,0,1]

#         elif(np.mod(pc_row_count, 8) == 2): # 2,10 ...
#             # print('encodecb_2', encode_cb_2)
#             if( (cloud_to_compare == encode_cb_2).all()):
#                 pass
#                 # print('**********condition exisits [0,1,0]')
#             else:
#                 changed_indices = np.where(cloud_to_compare != encode_cb_2)
#                 # print('2 change indices & length', changed_indices, len(changed_indices[0]))
#                 for i in range(len(changed_indices[0])):
#                     if(encode_cb_2[changed_indices[0][i]] == 1):
#                         cloud_row[changed_indices[0][i]] = cloud_row[changed_indices[0][i]] + 1
#                         cloud_row_halfdelta[changed_indices[0][i]] = cloud_row_halfdelta[changed_indices[0][i]] + 1
#                     else:
#                         cloud_row[changed_indices[0][i]] = cloud_row[changed_indices[0][i]] - 1
#                         cloud_row_halfdelta[changed_indices[0][i]] = cloud_row_halfdelta[changed_indices[0][i]] - 1

#                 # print('condition [101]', (count), cloud_row)
#             code_book = encode_cb_2 #[1,0,1]

#         elif(np.mod(pc_row_count, 8) == 3): # 3,11 ...
#             # print('encodecb_3', encode_cb_3)
#             if( (cloud_to_compare == encode_cb_3).all()):
#                 pass
#                 # print('**********condition exisits [0,1,1]')
#             else:
#                 changed_indices = np.where(cloud_to_compare != encode_cb_3)
#                 # print('3 change indices & length', changed_indices, len(changed_indices[0]))
#                 for i in range(len(changed_indices[0])):
#                     if(encode_cb_3[changed_indices[0][i]] == 1):
#                         cloud_row[changed_indices[0][i]] = cloud_row[changed_indices[0][i]] + 1
#                         cloud_row_halfdelta[changed_indices[0][i]] = cloud_row_halfdelta[changed_indices[0][i]] + 1
#                     else:
#                         cloud_row[changed_indices[0][i]] = cloud_row[changed_indices[0][i]] - 1
#                         cloud_row_halfdelta[changed_indices[0][i]] = cloud_row_halfdelta[changed_indices[0][i]] - 1

#                 # print('condition [011]', (count), cloud_row)
#             code_book = encode_cb_3 #[0,1,1]
        
#         elif(np.mod(pc_row_count, 8) == 4): # 4,12 ...
#             # print('encodecb_4', encode_cb_4)
#             if( (cloud_to_compare == encode_cb_4).all()):
#                 pass
#                 # print('**********condition exisits [1,0,0]')
#             else:
#                 changed_indices = np.where(cloud_to_compare != encode_cb_4)
#                 # print('4 change indices & length', changed_indices, len(changed_indices[0]))
#                 for i in range(len(changed_indices[0])):
#                     if(encode_cb_4[changed_indices[0][i]] == 1):
#                         cloud_row[changed_indices[0][i]] = cloud_row[changed_indices[0][i]] + 1
#                         cloud_row_halfdelta[changed_indices[0][i]] = cloud_row_halfdelta[changed_indices[0][i]] + 1
#                     else:
#                         cloud_row[changed_indices[0][i]] = cloud_row[changed_indices[0][i]] - 1
#                         cloud_row_halfdelta[changed_indices[0][i]] = cloud_row_halfdelta[changed_indices[0][i]] - 1

#                 # print('condition [100]', (count), cloud_row)
#             code_book = encode_cb_4 #[1,0,0]
        
#         elif(np.mod(pc_row_count, 8) == 5): # 5,13 ...
#             # print('encodecb_5', encode_cb_5)
#             if( (cloud_to_compare == encode_cb_5).all()):
#                 pass
#                 # print('**********condition exisits [1,0,1]')
#             else:
#                 changed_indices = np.where(cloud_to_compare != encode_cb_5)
#                 for i in range(len(changed_indices[0])):
#                     if(encode_cb_5[changed_indices[0][i]] == 1):
#                         cloud_row[changed_indices[0][i]] = cloud_row[changed_indices[0][i]] + 1
#                         cloud_row_halfdelta[changed_indices[0][i]] = cloud_row_halfdelta[changed_indices[0][i]] + 1
#                     else:
#                         cloud_row[changed_indices[0][i]] = cloud_row[changed_indices[0][i]] - 1
#                         cloud_row_halfdelta[changed_indices[0][i]] = cloud_row_halfdelta[changed_indices[0][i]] - 1

#                 # print('condition [101]', (count), cloud_row)
#             code_book = encode_cb_5 #[1,0,1]
        
#         elif(np.mod(pc_row_count, 8) == 6): # 6,14 ...
#             # print('encodecb_6', encode_cb_6)
#             if( (cloud_to_compare == encode_cb_6).all()):
#                 pass
#                 # print('**********condition exisits [1,1,0]')
#             else:
#                 changed_indices = np.where(cloud_to_compare != encode_cb_6)
#                 # print('6 change indices & length', changed_indices, len(changed_indices[0]))
#                 for i in range(len(changed_indices[0])):
#                     if(encode_cb_6[changed_indices[0][i]] == 1):
#                         cloud_row[changed_indices[0][i]] = cloud_row[changed_indices[0][i]] + 1
#                         cloud_row_halfdelta[changed_indices[0][i]] = cloud_row_halfdelta[changed_indices[0][i]] + 1
#                     else:
#                         cloud_row[changed_indices[0][i]] = cloud_row[changed_indices[0][i]] - 1
#                         cloud_row_halfdelta[changed_indices[0][i]] = cloud_row_halfdelta[changed_indices[0][i]] - 1

#                 # print('condition [110]', (count), cloud_row)
#             code_book = encode_cb_6 #[1,1,0]
        
#         elif(np.mod(pc_row_count, 8) == 7): # 7,15 ...
#             # print('encodecb_7', encode_cb_7)
#             if( (cloud_to_compare == encode_cb_7).all()):
#                 pass
#                 # print('**********condition exisits [1,1,1]')
#             else:
#                 changed_indices = np.where(cloud_to_compare != encode_cb_7)
#                 # print('7 change indices & length', changed_indices, len(changed_indices[0]))
#                 for i in range(len(changed_indices[0])):
#                     if(encode_cb_7[changed_indices[0][i]] == 1):
#                         cloud_row[changed_indices[0][i]] = cloud_row[changed_indices[0][i]] + 1
#                         cloud_row_halfdelta[changed_indices[0][i]] = cloud_row_halfdelta[changed_indices[0][i]] + 1
#                     else:
#                         cloud_row[changed_indices[0][i]] = cloud_row[changed_indices[0][i]] - 1
#                         cloud_row_halfdelta[changed_indices[0][i]] = cloud_row_halfdelta[changed_indices[0][i]] - 1

#                 # print('condition [111]', (count), cloud_row)
#             code_book = encode_cb_7 #[1,1,1]
        

#         quant_encoded.append(cloud_row)
#         quant_encoded_halfdelta.append(cloud_row_halfdelta)
#         cbook.append(code_book)
#         # print('row# and code book', pc_row_count, code_book)

#         pc_row_count = pc_row_count + 1
#         if(pc_row_count > len(pc_in)):
#             print('something wrong.. exceeded length of pc')
        
#         # pc_values_decoded.append(pc_values)
#         # cbook_decoded.append(cbook_from_pcvalues)

#         #get the updated numpyh representation, updated point cloud and the codebook from data
#         # cloud_row_numpyrep = cloud_row
#         # cloud_row_pcvalues = cloud_row_numpyrep*resolution + np.array([x[0],y[0],z[0]])
#         # codebook_decoded = np.array(get_oddflag(cloud_row_pcvalues))
#         # codebook_basedoncount = code_book

#     return quant_encoded, quant_encoded_halfdelta, cbook

def qim_quantize_restricted_threebits_new(pc_in):
    
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
    quant_encoded_halfdelta = []
    cbook = []
    # cbook_decoded = []
    # pc_values_decoded = []
    
    while pc_row_count < len(pc_in):
        cloud_to_compare = []
        cloud_row = []
        cloud_row_halfdelta = []

        pc = pc_in[pc_row_count, :]
        # cloud_row_clean = ((pc - np.array([x[0],y[0],z[0]])) / resolution).astype(np.int32)
        quantized_values_row = getQuantizedValues_from_pointCloud(pc, resolution_delta, x,y,z)
        # print('clean', cloud_row_clean)
        cloud_row =  quantized_values_row
        # print('input cloud-Delta', cloud_row)
        cloud_row_halfdelta =  2*cloud_row
        # print('input cloud- Half Delta', cloud_row_halfdelta)

        cloud_to_compare = np.array(get_oddflag(cloud_row))
        # print('input oddflag', cloud_to_compare)

        #reset code book
        code_book = [0,0,0]
        # print('cloud compare', cloud_to_compare)

        if(np.mod(pc_row_count, 8) == 0): # 0,8, 16...
            # print('encode_0', encode_cb_0)
            if( (cloud_to_compare == encode_cb_0).all()):
                # print('**********condition exists [0,0,0]')
                pass
            else:
                changed_indices = np.where(cloud_to_compare != encode_cb_0)
                # print('0 change indices & length', changed_indices, len(changed_indices[0]))
                # print('\n')
                for i in range(len(changed_indices[0])):
                        cloud_row_halfdelta[changed_indices[0][i]] = cloud_row_halfdelta[changed_indices[0][i]] + 1
                # print('condition [000]', (count), cloud_row)
            code_book = encode_cb_0 #[0,0,0]   
        
        elif(np.mod(pc_row_count, 8) == 1): # 1,9, ...
            # print('encodecb_1', encode_cb_1)
            if( (cloud_to_compare == encode_cb_1).all()):
                pass
                # print('**********condition exisits [0,0,1]')
            else:
                changed_indices = np.where(cloud_to_compare != encode_cb_1)
                # print('1 changed indices**', changed_indices)
                for i in range(len(changed_indices[0])):
                        cloud_row_halfdelta[changed_indices[0][i]] = cloud_row_halfdelta[changed_indices[0][i]] + 1
                
                # print('condition [001]', (count), cloud_row)
            code_book = encode_cb_1 #[0,0,1]

        elif(np.mod(pc_row_count, 8) == 2): # 2,10 ...
            # print('encodecb_2', encode_cb_2)
            if( (cloud_to_compare == encode_cb_2).all()):
                pass
                # print('**********condition exisits [0,1,0]')
            else:
                changed_indices = np.where(cloud_to_compare != encode_cb_2)
                # print('2 change indices & length', changed_indices, len(changed_indices[0]))
                for i in range(len(changed_indices[0])):
                        cloud_row_halfdelta[changed_indices[0][i]] = cloud_row_halfdelta[changed_indices[0][i]] + 1
                
                # print('condition [101]', (count), cloud_row)
            code_book = encode_cb_2 #[1,0,1]

        elif(np.mod(pc_row_count, 8) == 3): # 3,11 ...
            # print('encodecb_3', encode_cb_3)
            if( (cloud_to_compare == encode_cb_3).all()):
                pass
                # print('**********condition exisits [0,1,1]')
            else:
                changed_indices = np.where(cloud_to_compare != encode_cb_3)
                # print('3 change indices & length', changed_indices, len(changed_indices[0]))
                for i in range(len(changed_indices[0])):
                        cloud_row_halfdelta[changed_indices[0][i]] = cloud_row_halfdelta[changed_indices[0][i]] + 1
                
                # print('condition [011]', (count), cloud_row)
            code_book = encode_cb_3 #[0,1,1]
        
        elif(np.mod(pc_row_count, 8) == 4): # 4,12 ...
            # print('encodecb_4', encode_cb_4)
            if( (cloud_to_compare == encode_cb_4).all()):
                pass
                # print('**********condition exisits [1,0,0]')
            else:
                changed_indices = np.where(cloud_to_compare != encode_cb_4)
                # print('4 change indices & length', changed_indices, len(changed_indices[0]))
                for i in range(len(changed_indices[0])):
                        cloud_row_halfdelta[changed_indices[0][i]] = cloud_row_halfdelta[changed_indices[0][i]] + 1
                
                # print('condition [100]', (count), cloud_row)
            code_book = encode_cb_4 #[1,0,0]
        
        elif(np.mod(pc_row_count, 8) == 5): # 5,13 ...
            # print('encodecb_5', encode_cb_5)
            if( (cloud_to_compare == encode_cb_5).all()):
                pass
                # print('**********condition exisits [1,0,1]')
            else:
                changed_indices = np.where(cloud_to_compare != encode_cb_5)
                for i in range(len(changed_indices[0])):
                        cloud_row_halfdelta[changed_indices[0][i]] = cloud_row_halfdelta[changed_indices[0][i]] + 1
                
                # print('condition [101]', (count), cloud_row)
            code_book = encode_cb_5 #[1,0,1]
        
        elif(np.mod(pc_row_count, 8) == 6): # 6,14 ...
            # print('encodecb_6', encode_cb_6)
            if( (cloud_to_compare == encode_cb_6).all()):
                pass
                # print('**********condition exisits [1,1,0]')
            else:
                changed_indices = np.where(cloud_to_compare != encode_cb_6)
                # print('6 change indices & length', changed_indices, len(changed_indices[0]))
                for i in range(len(changed_indices[0])):
                        cloud_row_halfdelta[changed_indices[0][i]] = cloud_row_halfdelta[changed_indices[0][i]] + 1
                
                # print('condition [110]', (count), cloud_row)
            code_book = encode_cb_6 #[1,1,0]
        
        elif(np.mod(pc_row_count, 8) == 7): # 7,15 ...
            # print('encodecb_7', encode_cb_7)
            if( (cloud_to_compare == encode_cb_7).all()):
                pass
                # print('**********condition exisits [1,1,1]')
            else:
                changed_indices = np.where(cloud_to_compare != encode_cb_7)
                # print('7 change indices & length', changed_indices, len(changed_indices[0]))
                for i in range(len(changed_indices[0])):
                        cloud_row_halfdelta[changed_indices[0][i]] = cloud_row_halfdelta[changed_indices[0][i]] + 1
                

                # print('condition [111]', (count), cloud_row)
            code_book = encode_cb_7 #[1,1,1]
        

        quant_encoded.append(cloud_row)
        quant_encoded_halfdelta.append(cloud_row_halfdelta)
        cbook.append(code_book)
        # print('row# and code book', pc_row_count, code_book)

        pc_row_count = pc_row_count + 1
        if(pc_row_count > len(pc_in)):
            print('something wrong.. exceeded length of pc')
        
        # pc_values_decoded.append(pc_values)
        # cbook_decoded.append(cbook_from_pcvalues)

        #get the updated numpyh representation, updated point cloud and the codebook from data
        # cloud_row_numpyrep = cloud_row
        # cloud_row_pcvalues = cloud_row_numpyrep*resolution + np.array([x[0],y[0],z[0]])
        # codebook_decoded = np.array(get_oddflag(cloud_row_pcvalues))
        # codebook_basedoncount = code_book

    return quant_encoded, quant_encoded_halfdelta, cbook

def qim_quantize_restricted_twobits(pc_in):
    
    encode_cb_0 = np.array([0,0])
    encode_cb_1 = np.array([0,1])
    encode_cb_2 = np.array([1,0])
    encode_cb_3 = np.array([1,1])

    
    pc_row_count = 0

    quant_encoded = []
    quant_encoded_halfdelta = []
    cbook = []
    # cbook_decoded = []
    # pc_values_decoded = []
    
    while pc_row_count < len(pc_in):
        cloud_to_compare = []
        cloud_row = []
        cloud_row_halfdelta = []

        pc = pc_in[pc_row_count, :]
        # cloud_row_clean = ((pc - np.array([x[0],y[0],z[0]])) / resolution).astype(np.int32)
        quantized_values_row = getQuantizedValues_from_pointCloud(pc, resolution_delta, x,y,z)
        # print('clean', cloud_row_clean)
        cloud_row =  quantized_values_row[:2]
        # print('input cloud', cloud_row)
        cloud_row_halfdelta =  2*cloud_row
        # print('input cloud', cloud_row_halfdelta)

        cloud_to_compare = np.array(get_oddflag(cloud_row_halfdelta))
        # print('input oddflag', cloud_to_compare)

        #reset code book
        code_book = [0,0]
        # print('cloud compare', cloud_to_compare)

        if(np.mod(pc_row_count, 4) == 0): # 0,8, 16...
            # print('encode_0', encode_cb_0)
            if( (cloud_to_compare == encode_cb_0).all()):
                # print('**********condition exists [0,0,0]')
                pass
            else:
                changed_indices = np.where(cloud_to_compare != encode_cb_0)
                # print('0 change indices & length', changed_indices, len(changed_indices[0]))
                # print('\n')
                for i in range(len(changed_indices[0])):
                    if(encode_cb_0[changed_indices[0][i]] == 1):
                        cloud_row[changed_indices[0][i]] = cloud_row[changed_indices[0][i]] + 1
                        cloud_row_halfdelta[changed_indices[0][i]] = cloud_row_halfdelta[changed_indices[0][i]] + 1
                    else:
                        cloud_row[changed_indices[0][i]] = cloud_row[changed_indices[0][i]] - 1
                        cloud_row_halfdelta[changed_indices[0][i]] = cloud_row_halfdelta[changed_indices[0][i]] - 1

                # print('condition [000]', (count), cloud_row)
            code_book = encode_cb_0 #[0,0,0]   
        
        elif(np.mod(pc_row_count, 4) == 1): # 1,9, ...
            # print('encodecb_1', encode_cb_1)
            if( (cloud_to_compare == encode_cb_1).all()):
                pass
                # print('**********condition exisits [0,0,1]')
            else:
                changed_indices = np.where(cloud_to_compare != encode_cb_1)
                # print('1 changed indices**', changed_indices)
                for i in range(len(changed_indices[0])):
                    if(encode_cb_1[changed_indices[0][i]] == 1):
                        cloud_row[changed_indices[0][i]] = cloud_row[changed_indices[0][i]] + 1
                        cloud_row_halfdelta[changed_indices[0][i]] = cloud_row_halfdelta[changed_indices[0][i]] + 1
                    else:
                        cloud_row[changed_indices[0][i]] = cloud_row[changed_indices[0][i]] - 1
                        cloud_row_halfdelta[changed_indices[0][i]] = cloud_row_halfdelta[changed_indices[0][i]] - 1

                # print('condition [001]', (count), cloud_row)
            code_book = encode_cb_1 #[0,0,1]

        elif(np.mod(pc_row_count, 4) == 2): # 2,10 ...
            # print('encodecb_2', encode_cb_2)
            if( (cloud_to_compare == encode_cb_2).all()):
                pass
                # print('**********condition exisits [0,1,0]')
            else:
                changed_indices = np.where(cloud_to_compare != encode_cb_2)
                # print('2 change indices & length', changed_indices, len(changed_indices[0]))
                for i in range(len(changed_indices[0])):
                    if(encode_cb_2[changed_indices[0][i]] == 1):
                        cloud_row[changed_indices[0][i]] = cloud_row[changed_indices[0][i]] + 1
                        cloud_row_halfdelta[changed_indices[0][i]] = cloud_row_halfdelta[changed_indices[0][i]] + 1
                    else:
                        cloud_row[changed_indices[0][i]] = cloud_row[changed_indices[0][i]] - 1
                        cloud_row_halfdelta[changed_indices[0][i]] = cloud_row_halfdelta[changed_indices[0][i]] - 1

                # print('condition [101]', (count), cloud_row)
            code_book = encode_cb_2 #[1,0,1]

        elif(np.mod(pc_row_count, 4) == 3): # 3,11 ...
            # print('encodecb_3', encode_cb_3)
            if( (cloud_to_compare == encode_cb_3).all()):
                pass
                # print('**********condition exisits [0,1,1]')
            else:
                changed_indices = np.where(cloud_to_compare != encode_cb_3)
                # print('3 change indices & length', changed_indices, len(changed_indices[0]))
                for i in range(len(changed_indices[0])):
                    if(encode_cb_3[changed_indices[0][i]] == 1):
                        cloud_row[changed_indices[0][i]] = cloud_row[changed_indices[0][i]] + 1
                        cloud_row_halfdelta[changed_indices[0][i]] = cloud_row_halfdelta[changed_indices[0][i]] + 1
                    else:
                        cloud_row[changed_indices[0][i]] = cloud_row[changed_indices[0][i]] - 1
                        cloud_row_halfdelta[changed_indices[0][i]] = cloud_row_halfdelta[changed_indices[0][i]] - 1

                # print('condition [011]', (count), cloud_row)
            code_book = encode_cb_3 #[0,1,1]
        
        
        

        quant_encoded.append(cloud_row)
        quant_encoded_halfdelta.append(cloud_row_halfdelta)
        cbook.append(code_book)
        # print('row# and code book', pc_row_count, code_book)

        pc_row_count = pc_row_count + 1
        if(pc_row_count > len(pc_in)):
            print('something wrong.. exceeded length of pc')
        
        # pc_values_decoded.append(pc_values)
        # cbook_decoded.append(cbook_from_pcvalues)

        #get the updated numpyh representation, updated point cloud and the codebook from data
        # cloud_row_numpyrep = cloud_row
        # cloud_row_pcvalues = cloud_row_numpyrep*resolution + np.array([x[0],y[0],z[0]])
        # codebook_decoded = np.array(get_oddflag(cloud_row_pcvalues))
        # codebook_basedoncount = code_book

    return quant_encoded, quant_encoded_halfdelta, cbook


def qim_quantize_restricted_singlebit(pc_in):
    
    encode_cb_0 = np.array([0])
    encode_cb_1 = np.array([1])
    # encode_cb_2 = np.array([1,0])
    # encode_cb_3 = np.array([1,1])


    pc_row_count = 0

    quant_encoded = []
    quant_encoded_halfdelta = []
    cbook = []
    # cbook_decoded = []
    # pc_values_decoded = []
    
    while pc_row_count < len(pc_in):
        cloud_to_compare = []
        cloud_row = []
        cloud_row_halfdelta = []

        pc = pc_in[pc_row_count, :]
        # cloud_row_clean = ((pc - np.array([x[0],y[0],z[0]])) / resolution).astype(np.int32)
        quantized_values_row = getQuantizedValues_from_pointCloud(pc, resolution_delta, x,y,z)
        # print('clean', cloud_row_clean)
        cloud_row =  quantized_values_row[:1]
        # print('input cloud', cloud_row)
        cloud_row_halfdelta =  2*cloud_row
        # print('input cloud', cloud_row_halfdelta)

        cloud_to_compare = np.array(get_oddflag(cloud_row_halfdelta))
        # print('input oddflag', cloud_to_compare)

        #reset code book
        code_book = [0]
        # print('cloud compare', cloud_to_compare)

        if(np.mod(pc_row_count, 2) == 0): # 0,8, 16...
            # print('encode_0', encode_cb_0)
            if( (cloud_to_compare == encode_cb_0).all()):
                # print('**********condition exists [0,0,0]')
                pass
            else:
                changed_indices = np.where(cloud_to_compare != encode_cb_0)
                # print('0 change indices & length', changed_indices, len(changed_indices[0]))
                # print('\n')
                for i in range(len(changed_indices[0])):
                    if(encode_cb_0[changed_indices[0][i]] == 1):
                        cloud_row[changed_indices[0][i]] = cloud_row[changed_indices[0][i]] + 1
                        cloud_row_halfdelta[changed_indices[0][i]] = cloud_row_halfdelta[changed_indices[0][i]] + 1
                    else:
                        cloud_row[changed_indices[0][i]] = cloud_row[changed_indices[0][i]] - 1
                        cloud_row_halfdelta[changed_indices[0][i]] = cloud_row_halfdelta[changed_indices[0][i]] - 1

                # print('condition [000]', (count), cloud_row)
            code_book = encode_cb_0 #[0,0,0]   
        
        elif(np.mod(pc_row_count, 2) == 1): # 1,9, ...
            # print('encodecb_1', encode_cb_1)
            if( (cloud_to_compare == encode_cb_1).all()):
                pass
                # print('**********condition exisits [0,0,1]')
            else:
                changed_indices = np.where(cloud_to_compare != encode_cb_1)
                # print('1 changed indices**', changed_indices)
                for i in range(len(changed_indices[0])):
                    if(encode_cb_1[changed_indices[0][i]] == 1):
                        cloud_row[changed_indices[0][i]] = cloud_row[changed_indices[0][i]] + 1
                        cloud_row_halfdelta[changed_indices[0][i]] = cloud_row_halfdelta[changed_indices[0][i]] + 1
                    else:
                        cloud_row[changed_indices[0][i]] = cloud_row[changed_indices[0][i]] - 1
                        cloud_row_halfdelta[changed_indices[0][i]] = cloud_row_halfdelta[changed_indices[0][i]] - 1

                # print('condition [001]', (count), cloud_row)
            code_book = encode_cb_1 #[0,0,1]

        quant_encoded.append(cloud_row)
        quant_encoded_halfdelta.append(cloud_row_halfdelta)
        cbook.append(code_book)
        # print('row# and code book', pc_row_count, code_book)

        pc_row_count = pc_row_count + 1
        if(pc_row_count > len(pc_in)):
            print('something wrong.. exceeded length of pc')
        
        # pc_values_decoded.append(pc_values)
        # cbook_decoded.append(cbook_from_pcvalues)

        #get the updated numpyh representation, updated point cloud and the codebook from data
        # cloud_row_numpyrep = cloud_row
        # cloud_row_pcvalues = cloud_row_numpyrep*resolution + np.array([x[0],y[0],z[0]])
        # codebook_decoded = np.array(get_oddflag(cloud_row_pcvalues))
        # codebook_basedoncount = code_book

    return quant_encoded, quant_encoded_halfdelta, cbook

def compare_codebooks(encode_cb, decode_cb):
    print('encode cb length', encode_cb.shape[0])
    print('decode cb length', decode_cb.shape[0])
    

    if(encode_cb.shape[0] != decode_cb.shape[0]):
        print('***** size mismatch')

    point_match_count = 0
    mismatch_count = 0
    for i in range(encode_cb.shape[0]):
        if((encode_cb[i] == decode_cb[i]).all()):
            point_match_count += 1
        else:
            mismatch_count +=1
            print(i)

    print ('match count', point_match_count)
    print('mis-match count', mismatch_count)

        


if __name__ == '__main__':


    # 1. encode the point cloud and extract the codebook

    pc_input = pc_test
    

    print('point cloud shape and values', pc_input.shape, pc_input)
    # c = np.around(((pc_input - np.array([x[0],y[0],z[0]])) / resolution_delta)).astype(np.int32)
    # print('quantized pc numpy representation', c)

    # quantized_pc  = c*resolution_delta + np.array([x[0],y[0],z[0]])
    # print('pc values', quantized_pc) 

    voxel_delta, voxel_halfdelta, encoded_CB = qim_quantize_restricted_threebits(pc_input)
    voxel_halfdelta_npy = np.array([voxel_halfdelta]).reshape(-1,3)
    print('voxel_halfdelta_npy representation', voxel_halfdelta_npy)
    print('voxel delta', voxel_delta)
    # print('voxel_halfdelta', voxel_halfdelta)

    # 2. Get the PC from the quantized values 
    # #encoded_quantized_pc  = encoded_pc*resolution + np.array([x[0],y[0],z[0]])
    encoded_quantized_pc  = getPointCloud_from_quantizedValues(voxel_halfdelta_npy, resolution_halfdelta, x,y,z)
    print('qim pc values', encoded_quantized_pc) 
    # d = ((pc_input - np.array([x[0],y[0],z[0]])) / resolution).astype(np.int32)

    # # 3. Get the decoded codebook and quantized values of decoded point cloud   
    # # Decode the point cloud and get the code book
    
    decoded_CB, decoded_quantized_values = qim_decode(encoded_quantized_pc, resolution_halfdelta)
    print('decoded pc numpy representation', np.array([decoded_quantized_values]).reshape(-1,3))

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



