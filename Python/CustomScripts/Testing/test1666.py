import open3d as o3d
import os

# Create a Visualizer object
vis = o3d.visualization.Visualizer()
vis.create_window(width=800, height=800, window_name="PCD Visualization")

# Set the background colour
opt = vis.get_render_option()
opt.background_color = [1, 1, 1]  # White background
opt.point_size = 10

# Create a point cloud
base_path = r"E:\AILiveSim_1_9_7\SensorData\pcl"
pcd = o3d.io.read_point_cloud(os.path.join(base_path, "pcl0.pcd"))
pcd1 = o3d.geometry.PointCloud()
pcd1.points = o3d.utility.Vector3dVector([[0, 0, 0], [0.5, 0.5, 0], [1, 0, 0], [0, 1, 0]])

# Add the point cloud to the visualizer
vis.add_geometry(pcd)

# Create axis lines
# Define points for the lines
axis_points = [
    [0, 0, 0],  # Origin
    [100, 0, 0],  # X-axis direction
    [0, 100, 0],  # Y-axis direction
    [0, 0, 100],  # Z-axis direction
]

# Define lines connecting the origin to each axis direction
axis_lines = [
    [0, 1],  # X-axis
    [0, 2],  # Y-axis
    [0, 3],  # Z-axis
]

# Define colours for the axes: Red (X), Green (Y), Blue (Z)
axis_colours = [
    [1, 0, 0],  # Red for X-axis
    [0, 1, 0],  # Green for Y-axis
    [0, 0, 1],  # Blue for Z-axis
]

# Create a LineSet for the axes
line_set = o3d.geometry.LineSet()
line_set.points = o3d.utility.Vector3dVector(axis_points)
line_set.lines = o3d.utility.Vector2iVector(axis_lines)
line_set.colors = o3d.utility.Vector3dVector(axis_colours)

# Add the LineSet to the visualizer
vis.add_geometry(line_set)

# Adjust the camera
ctr = vis.get_view_control()
ctr.set_lookat([0, 0, 0])  # Focus on the origin
ctr.set_zoom(1.0)
ctr.set_front([1, 0, 0])  # Set front direction
ctr.set_up([0, 1, 0])     # Set upward direction

# Run the visualizer
vis.run()
