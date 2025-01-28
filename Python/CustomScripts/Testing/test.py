import threading, time, json
import ALSLib.TCPClient, ALSLib.ALSClient
import ALSLib.ALSHelperFunctionLibrary as ALSFunc
import ALSLib.ALSHelperImageLibrary as ALSImg
import ALSLib.ALSHelperLidarLibrary as ALSLidar
import torch
import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
import struct

# Implement LIDAR visualisation alongside the YOLO output window
HOST = '127.0.0.1'
ALSweights = "E:/Team1Docs/yolov5/weights_ailivesim/best.pt"
MixedWeights = "E:/Team1Docs/yolov5/weights_mixed/best.pt"

# Initialize YOLOv5 model
print("Loading YOLOv5 model...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torch.hub.load('ultralytics/yolov5', 'custom', path=ALSweights)
model.conf = 0.25  # Confidence threshold
model.iou = 0.45   # NMS IOU threshold

# Message handler for ALS client
def myMessageHandler(rawMessage):
    str = rawMessage.decode('utf-8')
    cmdList = str.split(" ")
    if cmdList[0].startswith("EndCondition"):
        TestContext.client.request_destroy_situation()
        TestContext.lock.acquire()
        TestContext.testEnded = True
        print("setting TestEnded")
        TestContext.lock.release()

class TestContext:
    lock = threading.Lock()
    testEnded = False    
    simulation_control_port = 9000
    client = ALSLib.ALSClient.Client((HOST, simulation_control_port), myMessageHandler)

# YOLO frame processing function
def process_frame(image):
    """
    Apply YOLOv5 detection on a single frame
    """
    if len(image.shape) == 2:  # If grayscale
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:  # If RGBA
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    
    results = model(image)
    processed_image = results.render()[0]  # Returns RGB image with detections
    
    # Convert to RGBA for consistency with StackImages
    processed_image = cv2.cvtColor(processed_image, cv2.COLOR_RGB2RGBA)
    return processed_image

# LiDAR point cloud serialization
def SerializeToPCLFileContent(numPoints, posX, posY, posZ, quatW, quatX, quatY, quatZ, point_array, timeEnd):
    posY = -posY  # Flip coordinates over y-axis to avoid mirrored display in o3d
    pclFileContent = '# .PCD v.7 - Point Cloud Data file format\nVERSION .7\nFIELDS x y z rgb time\n'\
                     'SIZE 4 4 4 4 4\nTYPE F F F U F\nCOUNT 1 1 1 1 1\nWIDTH %d\nHEIGHT 1\nVIEWPOINT %f %f %f %f %f %f %f\n'\
                     'POINTS %d\nDATA ascii\n' % (int(numPoints), posX, posY, posZ, quatW, quatX, quatY, quatZ, int(numPoints))
    
    for p in point_array:
        intensity = 1000
        if not math.isinf(p[3]):
            intensity = int(p[3])
        pclFileContent += '%.5f %.5f %.5f %d %.5f\n' % (p[0], p[1], p[2], intensity, timeEnd) 
    return pclFileContent

# Visualize LiDAR point cloud in 3D
def plot_ply(points):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    
    ax.scatter(x, y, z, c='r', marker='o')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    # Set fixed axis limits
    ax.set_xlim(-8000, 8000)
    ax.set_ylim(-8000, 8000)
    ax.set_zlim(-8000, 8000)

    # Equalize aspect ratio (workaround for 3D)
    def set_equal_aspect(ax):
        extents = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])
        centers = np.mean(extents, axis=1)
        ranges = np.max(extents[:, 1] - extents[:, 0])
        for ctr, rng, set_lim in zip(centers, [ranges] * 3, 
                                     [ax.set_xlim3d, ax.set_ylim3d, ax.set_zlim3d]):
            set_lim(ctr - rng / 2, ctr + rng / 2)

    set_equal_aspect(ax)
    plt.pause(0.07)
    plt.clf()

# Process LiDAR data and visualize
def process_pcloud(data):
    sizeofFloat = 4
    index = 11
    posX, posY, posZ, quatW, quatX, quatY, quatZ, numPoints, timeStart, timeEnd, numberOfBeams = struct.unpack('<fffffffffff', data[0:index * sizeofFloat])

    print('LiDAR: %f %f %f' % (posX, posY, posZ))

    pointCloudData = data[index * sizeofFloat:]
    point_array = np.frombuffer(pointCloudData, dtype=np.dtype("float32"))
    point_array = np.reshape(point_array, (-1, 4))

    # Plot LiDAR point cloud
    plot_ply(point_array)

    # Save point cloud to PCL file
    pclFileContent = SerializeToPCLFileContent(numPoints, posX, posY, posZ, quatW, quatX, quatY, quatZ, point_array, timeEnd)
    filename = f'output_{int(time.time())}.pcd'
    try:
        with open(filename, mode='w') as fileObject:
            fileObject.write(pclFileContent)
        print(f"Point cloud saved to {filename}")
    except Exception as e:
        print(f"Failed to save point cloud: {e}")

if __name__ == "__main__":
    TARGET_FPS = 10  # Set your desired frame rate
    FRAME_TIME = 1 / TARGET_FPS  # Time per frame in seconds
    
    # Initialize simulation
    TestContext.client.connect()
    TestContext.client.request_load_scenario('Default_Scenario')
    overrides = "EgoVehicleSettings\\DemoAllSensors.SensorProfile.ini;Sensors.Sensors.[0].StreamToNetwork;True"
    TestContext.client.request_load_situation_layer_with_overrides('DemoSensors', overrides)
    
    # Get the sensor list
    time.sleep(1)
    sensorlist = TestContext.client.get_sensor_list()
    parsed_json = json.loads(sensorlist)
    print("\n", parsed_json['sensors'][10])

    # Find the host and port of the camera
    for x in parsed_json['sensors']:
        if x['path'] == 'Sensors.[0]':
            sensor_cam = x
            break
    camera_port = sensor_cam['sensor_port']
    host = sensor_cam['sensor_ip']

    # Find the host and port of the LiDAR
    for y in parsed_json['sensors']:
        if y['path'] == 'Sensors.[1]':
            sensor_lidar = y
            break
    lidar_port = sensor_lidar['sensor_port']
    host = sensor_lidar['sensor_ip']

    print("Connecting sensor socket to  " + host + " " + str(camera_port))
    client = ALSLib.TCPClient.TCPClient(host, camera_port, 5)  # Camera connection
    client.connect(5)

    lidar_client = ALSLib.TCPClient.TCPClient(host, lidar_port, 5)  # LiDAR connection
    lidar_client.connect(5)

    imageNum = 0
    while imageNum < 200:
        frame_start_time = time.time()  # Start timing the frame

        # Read image from stream
        data = client.read()
        index = 0
        img, index, width, height = ALSFunc.ReadImage_Stream(data, index)
        processed_img = process_frame(img)

        ALSImg.JustDisplay(processed_img)

        # Read LiDAR data from stream
        lidar_data = lidar_client.read()
        process_pcloud(lidar_data)

        imageNum += 1
        
        # Break loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Frame rate control
        frame_time = time.time() - frame_start_time
        if frame_time < FRAME_TIME:
            time.sleep(FRAME_TIME - frame_time)

        # Optional: Print actual FPS
        actual_fps = 1.0 / (time.time() - frame_start_time)
        print(f"FPS: {actual_fps:.2f}", end='\r')

    cv2.destroyAllWindows()
