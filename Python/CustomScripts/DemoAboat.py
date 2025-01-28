import threading
import struct
import math
import json
import time
import numpy as np
import matplotlib.pyplot as plt
import ALSLib.TCPClient
import ALSLib.ALSClient 
import ALSLib.ALSHelperLidarLibrary as ALSLidar
from ALSLib.ALSHelperFunctionLibrary import get_sensordata_path
import ALSLib.ALSHelperFunctionLibrary as ALSFunc
import ALSLib.ALSHelperImageLibrary as ALSImg

#TODO

HOST = '127.0.0.1'
VL16 = "EgoVehicleSettings\DemoAllSensors.SensorProfile.ini"

def myMessageHandler(rawMessage):
    str = rawMessage.decode('utf-8')
    # print ('\n---> recieved Raw message: '+str)
    
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

"""def SerializeToPCLFileContent(numPoints, posX, posY, posZ, quatW, quatX, quatY, quatZ, point_array):

    posY = -posY
    pclFileContent = '# .PCD v.7 - Point Cloud Data file format\nVERSION .7\nFIELDS x y z rgb\n'\
    'SIZE 4 4 4 4\nTYPE F F F U\nCOUNT 1 1 1 1\nWIDTH %d\nHEIGHT 1\nVIEWPOINT %f %f %f %f %f %f %f\n'\
    'POINTS %d\nDATA ascii\n' % (int(numPoints), posX, posY, posZ, quatW, quatX, quatY, quatZ, int(numPoints))
    
    for p in point_array:
        intensity = 1000
        if not math.isinf(p[3]):
            intensity = int(p[3])
        pclFileContent += '%.5f %.5f %.5f %d\n' % (p[0], p[1], p[2], intensity) 
    return pclFileContent"""

"""fig = plt.figure()
def plot_ply(points):
    # Create or reuse the 3D subplot
    ax = fig.add_subplot(111, projection='3d')
    
    # Extract x, y, z coordinates
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    
    # Plot points
    ax.scatter(x, y, z, c='r', marker='o')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    # Set fixed axis limits
    ax.set_xlim(-10000, 10000)
    ax.set_ylim(-10000, 10000)
    ax.set_zlim(-10000, 10000)

    # Equalize aspect ratio (workaround for 3D)
    def set_equal_aspect(ax):
        extents = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])
        centers = np.mean(extents, axis=1)
        ranges = np.max(extents[:, 1] - extents[:, 0])
        for ctr, rng, set_lim in zip(centers, [ranges] * 3, 
                                     [ax.set_xlim3d, ax.set_ylim3d, ax.set_zlim3d]):
            set_lim(ctr - rng / 2, ctr + rng / 2)

    set_equal_aspect(ax)

    # Optional: Freeze limits on resizing (not always needed in 3D)
    def freeze_limits(event):
        ax.set_xlim(-10000, 10000)
        ax.set_ylim(-10000, 10000)
        ax.set_zlim(-10000, 10000)

    fig.canvas.mpl_connect('resize_event', freeze_limits)

    # Show the plot temporarily and clear
    plt.pause(0.07)
    plt.clf()
	#plt.show()"""	

if __name__ == "__main__":
    TestContext.client.connect()
    TestContext.client.request_load_scenario('Default_Scenario')
    overrides = "EgoVehicleSettings\ABoatWithSensors.SensorProfile.ini;Sensors.Sensors.[0].StreamToNetwork;True"
    TestContext.client.request_load_situation_layer_with_overrides('DemoABoat', overrides)

    #ALSLidar.create_sensor_data_folders()


    



    """sensorprofile_path = "EgoVehicleSettings\ABoatWithSensors.SensorProfile.ini"
    #sensor_path_in_file = "Sensors.Sensors.[1]"
    overrides = "EgoVehicleSettings\ABoatWithSensors.SensorProfile.ini;Sensors.Sensors[1].StreamToNetwork;True"
    #overrides = "{file_path};{sensor_path}.StreamToNetwork;True".format(file_path=sensorprofile_path, sensor_path=sensor_path_in_file)"
    TestContext.client.request_load_situation_layer_with_overrides('DemoABoat', overrides)
    #TestContext.client.request_load_situation_layer('DemoABoat')"""

    # Sensor list from which the camera is selected as defined in the sensor profile
    time.sleep(1)
    sensorlist = TestContext.client.get_sensor_list()
    parsed_json = json.loads(sensorlist)
    print(sensorlist)
    print(parsed_json)

        # Find the host and port of the camera
    for x in parsed_json['sensors']:
        if x['path'] == 'Sensors.[0]':
            sensor = x
            break
    camera_port = sensor['sensor_port']
    host = sensor['sensor_ip']

    print("host: ", host, "port: ", camera_port)

    print( "Connecting sensor socket to  " + host + " " + str(camera_port) )
    client = ALSLib.TCPClient.TCPClient(host, camera_port, 5)
    client.connect(5)
    
    imagenum = 0
    while(imagenum < 300):
        data = client.read()
        index = 0
        imagenum += 1
        img, index, width, height = ALSFunc.ReadImage_Stream(data, index)
        ALSImg.JustDisplay(img)
        """sizeofFloat = 4
        index = 11
        posX, posY, posZ, quatW, quatX, quatY, quatZ, numPoints, timeStart, timeEnd, \
        numberOfBeams = struct.unpack('<fffffffffff', data[0:index * sizeofFloat])
        print('PCL: %f %f %f', posX, posY, posZ)"""
        
        """# this code does the plot
        pointCloudData = data[index * sizeofFloat:]
        point_array = np.frombuffer(pointCloudData, dtype=np.dtype("float32"))
        point_array = np.reshape(point_array, (-1, 4))
        plot_ply(point_array)"""

        """# this code saves the point cloud to file
        pclFileContent = SerializeToPCLFileContent(numPoints, posX, posY, posZ, quatW, quatX, quatY, quatZ, point_array)"""
        
        """# Get the sensor data path
        filename = get_sensordata_path('/pcl/pcl' + str(imageNum) + '.pcd')
        
        try:
            with open(filename, mode='w') as fileObject:
                fileObject.write(pclFileContent)
            print(f"Point cloud saved to {filename}")
            imageNum += 1  # Increment the image number for the next file
        except Exception as e:
            print(f"Failed to save point cloud to file: {e}")"""
