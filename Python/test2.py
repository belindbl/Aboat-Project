import numpy as np
import open3d as o3d

# Create a point cloud with a few points
pcd = o3d.geometry.PointCloud()
points = np.array([[1, 1, 1], [1, 1, 2], [1, 1, 3]])
pcd.points = o3d.utility.Vector3dVector(points)

# Define the points for the origin and the end of each axis
axis_points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
lines = [[0, 1], [0, 2], [0, 3]]  # Lines from origin to each axis point
colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]  # Red for X, Green for Y, Blue for Z

# Create a LineSet for the axes
axis_lines = o3d.geometry.LineSet()
axis_lines.points = o3d.utility.Vector3dVector(axis_points)
axis_lines.lines = o3d.utility.Vector2iVector(lines)
axis_lines.colors = o3d.utility.Vector3dVector(colors)

o3d.visualization.draw_geometries([pcd, axis_lines])
'''
# Initialize the visualizer
vis = o3d.visualization.Visualizer()
vis.create_window(height=480, width=480)
vis.add_geometry(pcd)
vis.add_geometry(axis_lines)

# Run the visualizer with an exit condition
try:
    while vis.is_active():  # Check if the window is open
        vis.poll_events()
        vis.update_renderer()
except KeyboardInterrupt:
    pass
finally:
    vis.destroy_window()'''
