import numpy as np
import open3d as o3d
import time

#Create pcd
pcd = o3d.geometry.PointCloud()
points = np.array([[1, 1, 1], [1.1, 1, 2], [1, 1.1, 3]])
#points = np.random.rand(10,3)
print(points)
pcd.points = o3d.utility.Vector3dVector(points)

#Init visualizer
vis = o3d.visualization.Visualizer()
vis.create_window(height=480, width=480)
vis.add_geometry(pcd)

#Run visualizer

#while vis.is_active():
 #   vis.poll_events()
  #  vis.update_renderer()
   # time.sleep(15)
# Close the visualizer window if interrupted
vis.destroy_window()