# Author: Grégoire

import os
import numpy as np
from sklearn.cluster import DBSCAN
import open3d as o3d

def read_point_cloud_data(file_path, encoding='utf-8'):
    points = []
    try:
        with open(file_path, 'r', encoding=encoding) as file:
            for line in file:
                parts = line.strip().split()
                try:
                    point = [float(parts[0]), float(parts[1]), float(parts[2])]
                    points.append(point)
                except ValueError:
                    print(f"Skipping line: {line.strip()}")
    except UnicodeDecodeError:
        print(f"Error decoding file with {encoding} encoding. Trying ISO-8859-1 encoding.")
        with open(file_path, 'r', encoding='ISO-8859-1') as file:
            for line in file:
                parts = line.strip().split()
                point = [float(parts[0]), float(parts[1]), float(parts[2])]
                points.append(point)
    return np.array(points)

def filter_points(points, min_coords, max_coords):
    """Excludes points within a specific bounding box."""
    mask = ~(
        (points[:, 0] >= min_coords[0]) & (points[:, 0] <= max_coords[0]) & 
        (points[:, 1] >= min_coords[1]) & (points[:, 1] <= max_coords[1]) & 
        (points[:, 2] >= min_coords[2]) & (points[:, 2] <= max_coords[2])
    )
    return points[mask]

def get_coordinate_frame():
    return o3d.geometry.TriangleMesh.create_coordinate_frame(size=10)

def visualize_clusters(points, labels, save_image=False):
    # Create an Open3D point cloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    # Define custom colors for clusters
    custom_colors = {
        0: [1, 0, 0],    # Red    
        1: [0, 1, 0],    # Green  
        2: [0, 0, 1],    # Blue
        3: [1, 1, 0],    # Yellow
        4: [1, 0, 1],    # Magenta
        5: [0, 1, 1],    # Cyan   
        6: [0.5, 0, 0],  # Dark Red
        7: [0, 0.5, 0],  # Dark Green
        8: [0, 0, 0.5],  # Dark Blue
        9: [1, 0.75, 0], # Orange
        10: [0.4, 0.3, 0.1] # Brown
    }
    """
    without filter,
    Clusters 0, 1, and 5 are the åboat.
    """
    
    # Generate colors for any additional clusters
    unique_labels = len(set(labels))
    colors = np.random.rand(max(0, unique_labels - len(custom_colors)), 3)


    # Create color array for all points
    point_colors = np.zeros((len(points), 3))
    for i, label in enumerate(labels):
        if label == -1:
            point_colors[i] = [0, 0, 0]  # Black for noise
        else:
            if label < len(custom_colors):
                point_colors[i] = custom_colors[label]
            else:
                point_colors[i] = colors[label - len(custom_colors)]
    
    # Assign colors to point cloud
    pcd.colors = o3d.utility.Vector3dVector(point_colors)
    
    if save_image:
        # Create visualizer and save image
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        vis.add_geometry(pcd)
        vis.add_geometry(get_coordinate_frame())

        # Optimize view
        vis.get_render_option().point_size = 2.0
        vis.get_render_option().background_color = np.asarray([0.8, 0.8, 0.8])  # Light gray background
        
        # Adjust the camera 
        # Viewpont from above the boat, slighly behind with the boat looking straight ahead)
        ctr = vis.get_view_control()
        ctr.set_lookat([200, 0, 300])  # Focus on the origin (approximately where the LIDAR is located)
        ctr.set_zoom(0.02)
        ctr.set_front([-1, 0, 1])  # Set front direction
        ctr.set_up([1, 0, 1])     # Set upward direction

        # Let user adjust view
        vis.run()

        # Save screenshot
        vis.capture_screen_image("clusters.png")
        vis.destroy_window()
    else:
        # Direct visualization

        o3d.visualization.draw_geometries([pcd, get_coordinate_frame()])

    
def main():
    # Read data from file
    base_path = r"E:\AILiveSim_1_9_7\SensorData\pcl"
    file_path = os.path.join(base_path, "8.750.pcd")
    points = read_point_cloud_data(file_path)

    # Define the bounding box of the boat
    min_coords = np.array([-206, -159, 162])
    max_coords = np.array([513, 159, 321])
    
    # Filter out points inside the boat's bounding box
    filtered_points = filter_points(points, min_coords, max_coords)
    
    
    # DBSCAN parameters
    eps = 110 #Maximum distance between two points
    min_samples = 4  # Minimum points in neighborhood
    
    # Apply DBSCAN
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(filtered_points)
    
    # Print cluster information
    unique_labels = set(labels)
    n_clusters = len(unique_labels) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    
    print(f"Number of clusters: {n_clusters}")
    print(f"Number of noise points: {n_noise}")
    
    # Count points in each cluster
    for label in sorted(unique_labels):
        if label != -1:
            n_points = np.sum(labels == label)
            print(f"Cluster {label}: {n_points} points")
    
    # Visualize the clusters
    visualize_clusters(points, labels, save_image=True)  # Set save_image=False if you don't want to save the image


    print("test...\n")

    min_x= 99999
    max_x= -99999

    min_y= 99999
    max_y= -99999

    min_z= 99999
    max_z= -99999

    for point, label in zip(points, labels):
        if label == 5:  # Check if the point is in cluster 0
            print(f"Point {point} is in cluster {label}")

            # Get the minimum value of the x, y, or z coordinates for this specific point
            x_value = np.min(point[0])  # Minimum value of x-coordinates (for the current point)
            y_value = np.min(point[1])  # Minimum value of y-coordinates (for the current point)
            z_value = np.min(point[2])  # Minimum value of z-coordinates (for the current point)
            

            if(min_x>x_value):
                min_x=x_value
            if(min_y>y_value):
                min_y=y_value
            if(min_z>z_value):
                min_z=z_value

            if(max_x<x_value):
                max_x=x_value
            if(max_y<y_value):
                max_y=y_value
            if(max_z<z_value):
                max_z=z_value

            
            print("Minimum x is:", min_x)
            print("Minimum y is:", min_y)
            print("Minimum z is:", min_z)

            print("Maximum x is:", max_x)
            print("Maximum y is:", max_y)
            print("Maximum z is:", max_z)

            """
            Cluster 0:
            Minimum x is: -53.96654
            Minimum y is: -159.66811
            Minimum z is: 162.7607
            Maximum x is: 127.73042
            Maximum y is: 159.55305
            Maximum z is: 210.73123

            Cluster 1:
            Minimum x is: -206.80943
            Minimum y is: -112.12578
            Minimum z is: 202.25388
            Maximum x is: -111.3632
            Maximum y is: 113.82035
            Maximum z is: 321.6539

            Cluster 5:
            Minimum x is: 367.39731
            Minimum y is: -108.18796
            Minimum z is: 164.66702
            Maximum x is: 513.19049
            Maximum y is: 108.29727
            Maximum z is: 174.6118

            
            so, the aboat coordinates must be between
            min x: -206
            min y: -159
            min z: 162

            max x: 513
            max y: 159
            max z: 321

            TODO: excludes points from this area from clusterization
            """
    

if __name__ == "__main__":
    main()

