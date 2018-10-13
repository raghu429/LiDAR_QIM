## This project displays the KittiLiDAR data and computes clustering - This is the initial commit of the roject
* The data folder has multiple kitti LiDAR point clouds that the clustering algorithm would run on and produces the bounding box indices of those clusters.
* helper functions contain all the helper functions tath deal with the point clouds
* kitti_clustering hs the code that does the clustering using eucedian distance. It has code to extract multiple clusters from the LiDAR data and display them in rviz. Use the rviz config file in the rviz_config directory
* pointcloud_reader.py would provide functions to read the point cloud, understand the voxelization, camera filtering and the quantization parts.


### branch quantizationwatermarking_correlationtests
* This branch sets up code to perform watermarking and correlation tests in the file watermarking_correlation.py


## notes on clustering and kd trees
* At the tx and rx end
    get clusters and make np array of bounding boxes (index X rows X columns). Where index or the axis 0 is the number of the cluster
    populate the cluster_centeroid = centeroid of the clusters ( index X rows X column). Here, rows = 1, columns = 3
    filter the centeroids based on the location (your zone of interest) and update the new cluster centeroids. 
    Add Gaussian noise at the z-axis of the clusters that are filtered and here we could increase the sigma based on the distance (x)

* decoding steps
    first compare the lengths of the clusters for both tx and rcv files.
    convert cluster_centeroid points to kd tree to make the search optimal (optional)
    starting with the largest cluster_centeroid value (length wise) start getting the distance between corresponding points in the clusters and if it falls within a threshold and the indices match. Else from that index to the end of the indices.. compare the cluster with correlation and figure uut where the tampering occurred.

    