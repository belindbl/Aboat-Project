import numpy as np
from sklearn.cluster import DBSCAN

# Sample data (replace this with your actual point cloud data)
points = np.array([                 ##copy pasted from pcd391
    [82.23228, -0.70524, 179.52826, 316],
    [68.55805, -0.70877, 181.15465, 462],
    [64.26118, -0.70167, 185.49702, 457],
    [59.77107, -0.70386, 190.01311, 453],
    [55.06274, -0.67799, 194.74133, 448],
    [50.11564, -0.70003, 199.74532, 443],
    [44.91299, -0.68066, 205.04257, 438],
    [39.42727, -0.67385, 210.57465, 432],
    [-152.51479, -0.73993, 240.90520, 419],
    [-181.78784, -0.77870, 269.37943, 487],
    [-148.58179, -0.74095, 277.29700, 284],
    [85.82808, -99.04002, 171.73784, 316],
    [66.25352, -113.25954, 171.88005, 308],
    [14.99139, -150.55911, 165.79623, 389],
    [809.67218, -1805.80615, 92.51068, 472],
    [270.75189, -0.61838, 203.49915, 314],
    [413.81921, -0.59441, 171.93808, 309],
    [445.39777, -0.56928, 172.24861, 301],
    [856.40662, -0.43986, 112.16087, 323],
    [843.61517, -0.58843, 135.06918, 475],
    [842.77863, -0.59518, 155.64281, 485],
    [854.37579, -0.53865, 175.48235, 423],
    [1064.48840, -0.52243, 189.28616, 497],
    [369.38248, 106.99403, 167.59215, 313],
    [759.15790, 390.22760, 106.86444, 429],
    [758.98145, 390.25662, 129.24881, 467],
    [758.12140, 389.61035, 151.32169, 461],
    [782.10968, 406.91489, 171.31148, 278],
    [796.90485, 417.63428, 193.39720, 470],
    [796.05084, 417.10849, 216.81233, 470],
    [795.00360, 416.31381, 240.04791, 470],
    [794.11682, 415.70218, 263.16791, 470],
    [793.25861, 414.98386, 286.30304, 470],
    [781.72632, 406.79193, 307.91440, 363],
    [864.78076, 467.04254, 348.09033, 484],
    [866.07166, 467.96478, 375.26523, 419],
    [908.89545, 499.24301, 415.05679, 423],
    [84.95839, 97.68015, 171.70677, 316],
    [65.30766, 111.91529, 171.85347, 308],
    [12.30096, 150.44398, 165.38138, 389],
    [-1407.10095, 1181.16248, 84.57859, 398]

])




# DBSCAN parameters
eps = 5  # Maximum distance between two points to be considered in the same neighborhood
min_samples = 5  # Minimum number of points to form a dense region (cluster)

# Apply DBSCAN
dbscan = DBSCAN(eps=eps, min_samples=min_samples)
labels = dbscan.fit_predict(points)

# Print the results
for point, label in zip(points, labels):
    print(f"Point {point} is in cluster {label}")

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
plt.show()
