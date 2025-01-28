# Author: Benjamin Lindblom

import open3d as o3d
import os

# Visualiser for point cloud objects, after running the script, it will ask the user to input the file name
# of the point cloud. Do not give the file extension, only the name. E.g., "8" or "4.326" NOT "4.326.pcd"

# Create a Visualizer object
vis = o3d.visualization.Visualizer()
vis.create_window(width=1200, height=900, window_name="PCD Visualization")

# Set the background colour
opt = vis.get_render_option()
opt.background_color = [1, 1, 1]  # White background
opt.point_size = 3.5

# Create a point cloud
base_path = r"E:\AILiveSim_1_9_7\SensorData\pcl"
#base_path= "E:\AILiveSim_1_9_7\SensorData\LidarData_Default_Scenario\override_0"

# Get the PCD file name from the user
pcd_input = input("Enter the PCD file name: ")
#pcd = o3d.io.read_point_cloud(os.path.join(base_path, f"pcl{pcd_input}.pcd"))
#fawaz- commented previous line and added this
pcd = o3d.io.read_point_cloud(os.path.join(base_path, f"{pcd_input}.pcd"))
#pcd = o3d.io.read_point_cloud(os.path.join(base_path, f"LID_{pcd_input}.pcd"))

# Add the point cloud to the visualizer
vis.add_geometry(pcd)

# Create axis lines
# Define points for the lines
axis_points = [
    [0, 0, 0],  # Origin
    [10000, 0, 0],  # X-axis direction, X-axis extended to compare captured images to pcds.
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
# Viewpont from above the boat, slighly behind with the boat looking straight ahead)
ctr = vis.get_view_control()
ctr.set_lookat([200, 0, 300])  # Focus on the origin (approximately where the LIDAR is located)
ctr.set_zoom(0.02)
ctr.set_front([-1, 0, 1])  # Set front direction
ctr.set_up([1, 0, 1])     # Set upward direction
# Run the visualizer
vis.run()
vis.destroy_window()