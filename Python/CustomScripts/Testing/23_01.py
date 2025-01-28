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
        0: [1, 1, 1],  # White
        1: [1, 1, 1],  # White
        2: [0, 0, 1],  # Blue
        3: [1, 1, 0],  # Yellow
        4: [1, 0, 1],  # Magenta
        5: [1, 1, 1],  # White
        6: [0.5, 0, 0],  # Dark Red
        7: [0, 0.5, 0],  # Dark Green
        8: [0, 0, 0.5],  # Dark Blue
        9: [1, 0.75, 0], # Orange
        10: [0.4, 0.3, 0.1] # Brown
    }
    
    # Generate colors for additional clusters
    unique_labels = len(set(labels))
    colors = np.random.rand(max(0, unique_labels - len(custom_colors)), 3)

    # Assign colors to points
    point_colors = np.zeros((len(points), 3))
    for i, label in enumerate(labels):
        if label == -1:
            point_colors[i] = [0, 0, 0]  # Black for noise
        else:
            if label < len(custom_colors):
                point_colors[i] = custom_colors[label]
            else:
                point_colors[i] = colors[label - len(custom_colors)]
    
    pcd.colors = o3d.utility.Vector3dVector(point_colors)
    
    if save_image:
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        vis.add_geometry(pcd)
        vis.add_geometry(get_coordinate_frame())

        # Adjust the camera 
        # Viewpont from above the boat, slighly behind with the boat looking straight ahead)
        ctr = vis.get_view_control()
        ctr.set_lookat([200, 0, 300])  # Focus on the origin (approximately where the LIDAR is located)
        ctr.set_zoom(0.02)
        ctr.set_front([-1, 0, 1])  # Set front direction
        ctr.set_up([1, 0, 1])     # Set upward direction

        vis.run()
        vis.destroy_window()
    else:
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
    eps = 110  # Maximum distance between two points
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
    visualize_clusters(filtered_points, labels, save_image=True)

if __name__ == "__main__":
    main()
