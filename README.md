## This project displays the KittiLiDAR data and computes clustering
* The data folder has multiple kitti LiDAR point clouds that the clustering algorithm would run on and produces the bounding box indices of those clusters.
* helper functions contain all the helper functions tath deal with the point clouds
* kitti_clustering hs the code that does the clustering using eucedian distance. It has code to extract multiple clusters from the LiDAR data and display them in rviz. Use the rviz config file in the rviz_config directory
* pointcloud_reader.py would provide functions to read the point cloud, understand the voxelization, camera filtering and the quantization parts.
