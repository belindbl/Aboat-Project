import os
import numpy as np
from sklearn.cluster import DBSCAN

# Function to read point cloud data from a file with a specified encoding
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

# Read data from file
base_path = r"E:\AILiveSim_1_9_7\SensorData\pcl"
file_path = os.path.join(base_path, "8.750.pcd")
points = read_point_cloud_data(file_path)

# DBSCAN parameters
eps = 110  # Maximum distance between two points to be considered in the same neighborhood
min_samples = 4  # Minimum number of points to form a dense region (cluster)

# Apply DBSCAN
dbscan = DBSCAN(eps=eps, min_samples=min_samples)
labels = dbscan.fit_predict(points)

# Print the results
for point, label in zip(points, labels):
    print(f"Point {point} is in cluster {label}")
"""
# Example visualization (optional)
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
unique_labels = set(labels)
colors = plt.cm.get_cmap("Spectral")(np.linspace(0, 1, len(unique_labels)))

for label, color in zip(unique_labels, colors):
    class_member_mask = (labels == label)
    xyz = points[class_member_mask]
    ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=[color], label=f'Cluster {label}')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.legend()
plt.show()"""
