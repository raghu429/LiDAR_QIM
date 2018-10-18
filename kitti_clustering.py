#!/usr/bin/env python
import sys
import numpy as np
import pcl
import rospy
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2, PointField
from random import randint
import struct
import ctypes
import std_msgs.msg
from helper_functions import *
from tamper_pc import *



def pcl_to_ros(pcl_array):
    """ Converts a pcl PointXYZRGB to a ROS PointCloud2 message
    
        Args:
            pcl_array (PointCloud_PointXYZRGB): A PCL XYZRGB point cloud
            
        Returns:
            PointCloud2: A ROS point cloud
    """
    ros_msg = PointCloud2()

    ros_msg.header.stamp = rospy.Time.now()
    ros_msg.header.frame_id = "velodyne"

    ros_msg.height = 1
    ros_msg.width = pcl_array.size

    ros_msg.fields.append(PointField(
                            name="x",
                            offset=0,
                            datatype=PointField.FLOAT32, count=1))
    ros_msg.fields.append(PointField(
                            name="y",
                            offset=4,
                            datatype=PointField.FLOAT32, count=1))
    ros_msg.fields.append(PointField(
                            name="z",
                            offset=8,
                            datatype=PointField.FLOAT32, count=1))
    ros_msg.fields.append(PointField(
                            name="rgb",
                            offset=16,
                            datatype=PointField.FLOAT32, count=1))

    ros_msg.is_bigendian = False
    ros_msg.point_step = 32
    ros_msg.row_step = ros_msg.point_step * ros_msg.width * ros_msg.height
    ros_msg.is_dense = False
    buffer = []

    for data in pcl_array:
        s = struct.pack('>f', data[3])
        i = struct.unpack('>l', s)[0]
        pack = ctypes.c_uint32(i).value

        r = (pack & 0x00FF0000) >> 16
        g = (pack & 0x0000FF00) >> 8
        b = (pack & 0x000000FF)

        buffer.append(struct.pack('ffffBBBBIII', data[0], data[1], data[2], 1.0, b, g, r, 0, 0, 0, 0))

    ros_msg.data = "".join(buffer)

    return ros_msg

def rgb_to_float(color):
    """ Converts an RGB list to the packed float format used by PCL
    
        From the PCL docs:
        "Due to historical reasons (PCL was first developed as a ROS package),
         the RGB information is packed into an integer and casted to a float"
    
        Args:
            color (list): 3-element list of integers [0-255,0-255,0-255]
            
        Returns:
            float_rgb: RGB value packed as a float
    """
    hex_r = (0xff & color[0]) << 16
    hex_g = (0xff & color[1]) << 8
    hex_b = (0xff & color[2])

    hex_rgb = hex_r | hex_g | hex_b

    float_rgb = struct.unpack('f', struct.pack('i', hex_rgb))[0]

    return float_rgb

def random_color_gen():
    """ Generates a random color
    
        Args: None
        
        Returns: 
            list: 3 elements, R, G, and B
    """
    r = randint(0, 255)
    g = randint(0, 255)
    b = randint(0, 255)
    return [r, g, b]

def get_color_list(cluster_count):
    """ Returns a list of randomized colors
    
        Args:
            cluster_count (int): Number of random colors to generate
            
        Returns:
            (list): List containing 3-element color lists
    """
    if (cluster_count > len(get_color_list.color_list)):
        for i in xrange(len(get_color_list.color_list), cluster_count):
            get_color_list.color_list.append(random_color_gen())
    return get_color_list.color_list

def get_clusters(cloud, tolerance, min_size, max_size):

  tree = cloud.make_kdtree()
  extraction_object = cloud.make_EuclideanClusterExtraction()

  extraction_object.set_ClusterTolerance(tolerance)
  extraction_object.set_MinClusterSize(min_size)
  extraction_object.set_MaxClusterSize(max_size)
  extraction_object.set_SearchMethod(tree)

  # Get clusters of indices for each cluster of points, each clusterbelongs to the same object
  # 'clusters' is effectively a list of lists, with each list containing indices of the cloud
  clusters = extraction_object.Extract()
  return clusters
  

# clusters is a list of lists each list containing indices of the cloud
# cloud is an array with each cell having three numbers corresponding to x, y, z position
# Returns list of [x, y, z, color]
def get_colored_clusters(clusters, cloud):
  
  # Get a random unique colors for each object
  number_of_clusters = len(clusters)
  colors = get_color_list(number_of_clusters)

  colored_points = []

  # Assign a color for each point
  # Points with the same color belong to the same cluster
  #by doing an enumerate on the list of list (clusters) we are adding a counter variable to each cluster
  #in the clusters
  print(' number of elements in the cloud', cloud.size)
  print('cloud', cloud[7], cloud[8], cloud[9])
  #print('cloud elements 7,8,9,10', cloud[7, :], cloud[8, :], cloud[9, :], cloud[10, :])
  for cluster_id, cluster in enumerate(clusters):
    #print ('cluster id and length and cluster', cluster_id, len(cluster), cluster)
    for c, i in enumerate(cluster):
      x, y, z = cloud[i][0], cloud[i][1], cloud[i][2]
      color = rgb_to_float(colors[cluster_id])
      colored_points.append([x, y, z, color])
  
  return colored_points

def get_clustercorners(clusters, cloud):
    corners = []
    
    for cluster_id, cluster in enumerate(clusters):
        cluster_point_list = []
        #print ('cluster id and length and cluster', cluster_id, len(cluster), cluster)
        for c, i in enumerate(cluster):
             x, y, z = cloud[i][0], cloud[i][1], cloud[i][2]
             cluster_point_list = np.append(cluster_point_list, [x, y, z])
             #print('cluster: %s values:%s', c, [x,y,z])
        #print('cluster points list', cluster_point_list)
        cluster_point_list = cluster_point_list.reshape(-1, 3)
        #print('cluster points list & shape', cluster_point_list, cluster_point_list.shape)
        #print('cluster points shape', cluster_point_list.shape)
        x_min = cluster_point_list[:,0].min()
        y_min = cluster_point_list[:,1].min()
        z_min = cluster_point_list[:,2].min()

        x_max = cluster_point_list[:,0].max()
        y_max = cluster_point_list[:,1].max()
        z_max = cluster_point_list[:,2].max()

        print('cluter point list min', x_min, y_min, z_min )
        print('cluter point list min', x_max, y_max, z_max )

        corner = get_boundingboxcorners([x_min, y_min, z_min], [x_max, y_max, z_max])
        print('corner shape & value', corner.shape, corner)
        corners = np.append(corners, [corner])
    #print('corner shape and values', corners.shape, corners)
    return corners

def get_boundingboxcorners(point_min, point_max):
    print('point min', point_min)    
    print('point max', point_max)
    x, y, z = point_min
    print('x,y,z, min', x,y,z)
    xx, yy, zz = point_max
    print('x,y,z, max', xx,yy,zz)
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


def load_pc_from_pcd(pcd_path):
    """Load PointCloud data from pcd file."""
    p = pcl.load(pcd_path)
    return np.array(list(p), dtype=np.float32)

def kitti_cluster(pc_in):
 
  pcl_data = pcl.PointCloud()
  pcl_data.from_list(pc_in)
  colorless_cloud = pcl_data
  print ('cloud data', np.size(colorless_cloud))
  
  # Get groups of indices for each cluster of points
  # Each group of points belongs to the same object
  # This is effectively a list of lists, with each list containing indices of the cloud
  clusters = get_clusters(pcl_data, tolerance = 0.8, min_size = 20, max_size = 1500)
  #clusters = get_clusters(pcl_data, tolerance = 0.8, min_size = 20, max_size = 35)
  print('clusters found', len(clusters))

  #print('clusters', clusters)

  # Assign a unique color float for each point (x, y, z)
  # Points with the same color belong to the same cluster
  colored_points = get_colored_clusters(clusters, colorless_cloud)
  print('colored points length', len(colored_points))

  cluster_corners_msg = get_clustercorners(clusters, colorless_cloud)
  cluster_corners_msg = cluster_corners_msg.reshape(-1, 8,3)
  print('corners shape', cluster_corners_msg.shape)

  # Create a cloud with each cluster of points having the same color
  clusters_cloud = pcl.PointCloud_PointXYZRGB()
  clusters_cloud.from_list(colored_points)

  print('clusters cloud len', clusters_cloud)

  # Convert pcl data to ros messages
  #objects_msg = pcl_to_ros(objects_cloud)
  #table_msg = pcl_to_ros(table_cloud)
  
  clusters_msg = pcl_to_ros(clusters_cloud)

  return clusters_msg, cluster_corners_msg

  #print('cluster msg', clusters_msg)
  
  # Publish ROS messages
  #objects_publisher.publish(objects_msg)
  #table_publisher.publish(table_msg)
  #clusters_publisher.publish(clusters_msg)

def get_clustercenteroid(cluster_corner_list):
    #make sure that the shape is what you expect index * rows (8) * columns (3)
    cluster_corner_list = np.reshape(cluster_corner_list,(-1, 8, 3))
    print('cluster_corner_list shape', cluster_corner_list.shape)
    cluster_centeroid_list = np.array([[np.average(cluster_corner_list[i,:,0]), np.average(cluster_corner_list[i,:,1]), np.average(cluster_corner_list[i,:,2])] for i in range(0, cluster_corner_list.shape[0])])
    
    print ('cluster centeroid shape and value', cluster_centeroid_list.shape, cluster_centeroid_list)

    return cluster_centeroid_list

def get_kdtree_ofpc(points):
    #In this function since we try to get a kd-tree from point cloud library we need to convert the input numpy array into point cloud
    pc_points = pcl.PointCloud(points)
    kdtree_points = pc_points.make_kdtree_flann()

    return(kdtree_points)
  
#Compare the cluster centeroids
def get_clustercenteroid_changeindex(reference_list, modified_list, threshold):
    
    #determine the lengths of point clouds
    reference_len = reference_list.shape[0]
    modified_len = modified_list.shape[0]
    
    #make a kd tree of reference list
    kdtree_reference_centeroids = get_kdtree_ofpc(reference_list.astype(np.float32))

    #make the point cloud reference to the modified list
    modified_points = pcl.PointCloud(modified_list.astype(np.float32))

    #find the closest point index and the distance
    indices, sqr_distances = kdtree_reference_centeroids.nearest_k_search_for_cloud(modified_points, 1)
    
    suspect_indices = []
    modified_list_missing_indices =[]
    nonsuspect_indices = []

    print('indices and distances', indices, sqr_distances)

    for i in range(0, reference_len):
        print('index of the closest point in reference_list to point %d in modified_list is %d' % (i, indices[i, 0]))
        print('the squared distance between these two points is %f' % sqr_distances[i, 0])
        if(sqr_distances[i, 0] > threshold):
            suspect_indices.append(i)
        else:
            nonsuspect_indices.append(i)

    print('suspect indices', suspect_indices)
    print('non suspect indices', nonsuspect_indices)
    
    modified_list_indices  = np.arange(modified_len)
    
    #check which indices are missing in the modified list
    modified_list_missing_set = set(nonsuspect_indices).symmetric_difference(set(modified_list_indices))

    print('missing indices set length', len(list(modified_list_missing_set)))
    #retrieve elements from the set
    for i in range(0, len(list(modified_list_missing_set))):
        modified_list_missing_indices.append(list(modified_list_missing_set)[i])
        print('modified_list_missing_index', list(modified_list_missing_set)[i])
    
    #concatenate two lists
    suspect_cluster_indices = modified_list_missing_indices + suspect_indices
    print('final suspect indices', suspect_cluster_indices) 

    return suspect_cluster_indices


def linearcorrelation_comparison(mean, variance, noise_length, point_cloud, axis):
    reference_noise = np.random.normal(mean, variance, noise_length)
    # Do the correlation
    lcs = np.correlate(reference_noise, point_cloud[:, axis])
    correlation_value = (lcs/noise_length) 
    print('lcs', correlation_value)
    return (correlation_value)


if __name__ == '__main__':

 # ROS node initialization
  rospy.init_node('clustering_kitti', anonymous = True)
  
  # Initialize color_list
  get_color_list.color_list = []

 # Create Publishers
  clusters_publisher = rospy.Publisher("/pcl_clusters", PointCloud2, queue_size = 10000)

  # read pcl and the point clid
  #kitti_cluster("./data/002937.bin")
  # Get a point cloud of only the position information without color information
#  velodyne_path = "./data/000002.bin"
#  velodyne_path = "./data/000003.bin"
#  velodyne_path = "./data/000004.bin"
#  velodyne_path = "./data/000005.bin"
#  velodyne_path = "./data/000006.bin"
  #velodyne_path = "./data/002937.bin"
  velodyne_path = "./data/002937_addedcar.pcd"

# read the point cloud from pcd file
  dataformat = 'pcd'
  if dataformat == "bin":
    points_list = load_pc_from_bin(velodyne_path)
  elif dataformat == "pcd":
    points_list = load_pc_from_pcd(velodyne_path)

#points_list = load_pc_from_bin(velodyne_path)
#convert the point cloud list into a pcl xyz format

  resolution = 0.2
  voxel_shape=(800, 800, 40)
  x=(0, 80)
  y=(-40, 40)
  z=(-2.5, 1.5)
    
  #pc_logicalbounds = apply_logicalbounds_topc(points_list, x=x, y=y, z=z)
  logical_bounds = get_pc_logicaldivision(points_list, x=x, y=y, z=z)
  pc_logicalbounds = apply_logicalbounds_topc(points_list, logical_bounds)
  #In the camera angle filter function we remove the ground  plane by doing z thresholding
  pc_camera_angle_filtered  = filter_camera_angle_groundplane(points_list[:,:3])

  cluster_cloud_color, cluster_corners = kitti_cluster(pc_camera_angle_filtered)
  cluster_centeroids = get_clustercenteroid(cluster_corners) #here we get a flat (288,) element point array this could be reshaped to (12,8,3) i,e 12 corners with (8,3) or (96,3) i,e 96  x,y,z, points

  #print('cluster centeroid[:, 0]', cluster_centeroids[:, 0] )
  #these logical bounds which si a flat array of rue or False of the size of number of clusters can be used to extract corners from the cluster_corners in the next step.
  cluster_logical_bounds = get_pc_logicaldivision(cluster_centeroids.reshape(-1,3), x=(0,80), y=(-5, 5), z=(-1.5,1.5))
  
  print('cluster logical bounds', cluster_logical_bounds)
  cluster_corners_filtered = cluster_corners.reshape(-1,8,3)[cluster_logical_bounds]
  print('cluster corneres filtered shape', cluster_corners_filtered.shape)
  
  filtered_cluster_centeroids = apply_logicalbounds_topc(cluster_centeroids.reshape(-1,3), cluster_logical_bounds)
  print('filtered centeroids shape & values', filtered_cluster_centeroids.shape, filtered_cluster_centeroids)

  #this is an optional step and can be removed if needed later 
  #kdtree_centerids = get_kdtree_ofpc(filtered_cluster_centeroids)

  # Here the encoding starts
  min_sigma = 0.1
  max_sigma  = 0.5
  gap = (max_sigma - min_sigma)/filtered_cluster_centeroids.shape[0]
  sigmalist = np.arange(min_sigma,max_sigma,gap)
  meanlist =  np.zeros(filtered_cluster_centeroids.shape[0])
  axis = 2
  global_std = 0.001
  global_mean = 0.0
  encoded_pointcoud = add_gaussianNoise_clusters(pc_camera_angle_filtered, filtered_cluster_centeroids, sigmalist, meanlist, axis, global_std, global_mean)

 
#   #decoding steps:
#   1. read the point cloud
#   2. run it through filter_camera_angle_groundplane
#   3. get the clusters and their cluster_centeroids
#   4. compare cluster centeroids to get the suspect cluster index
#   5. do the crrelation starting from the suspect cluster and find out the cluster where there is no correlation


  i = 0
  # Spin while node is not shutdown
  while not rospy.is_shutdown():
    # read pcl and the point clid
    publish_pc2(cluster_corners.reshape(-1,3)[:,:], "/pointcloud_clustercorners")
    publish_pc2(encoded_pointcoud.reshape(-1,3)[:,:], "/pointcloud_encoded")
    publish_pc2(cluster_centeroids.reshape(-1,3)[:,:], "/pointcloud_clustercorners_centeroids")
    publish_pc2(filtered_cluster_centeroids.reshape(-1,3)[:,:], "/pointcloud_clustercorners_centeroids_filtered")
    publish_pc2(cluster_corners_filtered.reshape(-1,3)[:,:], "/pointcloud_clustercorners_filtered")
    publish_pc2(pc_camera_angle_filtered, "/pointcloud_camerafiltered_clustering")
    clusters_publisher.publish(cluster_cloud_color)
    print('spinn count', i)
    i += 1
  rospy.spin()


