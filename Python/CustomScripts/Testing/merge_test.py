import threading
import struct
import math
import numpy as np
import os
import cv2
import ALSLib.TCPClient
import ALSLib.ALSClient 
import ALSLib.ALSHelperLidarLibrary as ALSLidar
from ALSLib.ALSHelperFunctionLibrary import get_sensordata_path


#GStreamer necessary for RTSP
HOST = '127.0.0.1'
SENSOR_PROFILE = "EgoVehicleSettings\DemoAllSensors.SensorProfile.ini"
OS1_64 = "EgoVehicleSettings\CustomLidar.SensorProfile.ini"

ABOAT = "EgoVehicleSettings\AboatSensorTest.SensorProfile.ini" #camera and lidar
TESTING_PROFILE = "E:\AILiveSim_1_9_7\ConfigFiles\Scenario\Testing.ini"

# ATTEMPTING TO GET PARALLEL OUTPUT FROM CAMERA AND LIDAR
# BASED ON THE SAME SITUATION AS "LiveSocketLidar.py"
# LIDAR DATA IS COLLECTIBLE, BUT THE CAMERA CONNECTION FAILS

# Camera image save path
CAMERA_SAVE_PATH = r"E:\AILiveSim_1_9_7\SensorData\imgs"
if not os.path.exists(CAMERA_SAVE_PATH):
    os.makedirs(CAMERA_SAVE_PATH)

# Global variables
MAX_LIDAR_PCDS = 300
lidar_pcd_count = 0
stop_threads_event = threading.Event()  # Event to signal threads to stop

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

def SerializeToPCLFileContent(numPoints, posX, posY, posZ, quatW, quatX, quatY, quatZ, point_array):
    pclFileContent = '# .PCD v.7 - Point Cloud Data file format\nVERSION .7\nFIELDS x y z rgb\n'\
    'SIZE 4 4 4 4\nTYPE F F F U\nCOUNT 1 1 1 1\nWIDTH %d\nHEIGHT 1\nVIEWPOINT %f %f %f %f %f %f %f\n'\
    'POINTS %d\nDATA ascii\n' % (int(numPoints), posX, posY, posZ, quatW, quatX, quatY, quatZ, int(numPoints))
    
    for p in point_array:
        intensity = 1000
        if not math.isinf(p[3]) and p[3] != 0:
            intensity = int(p[3])
        pclFileContent += '%.5f %.5f %.5f %d\n' % (p[0], -p[1], p[2], intensity) 
    return pclFileContent


#TODO Investigate why the camera feed is not working
# Camera thread to handle image display and saving
def camera_thread():
    cam_client = ALSLib.TCPClient.TCPClient(HOST, 8880, 5)  # Assuming camera is on port 8880
    
    try:
        cam_client.connect(5)  # Attempt to connect to the camera client
        print("Connected to the camera.")
    except Exception as e:
        print(f"Error: Unable to connect to the camera on port 8880. {e}")
        stop_threads_event.set()  # Stop all threads if connection fails
        return
    
    image_num = 0
    while not stop_threads_event.is_set():  # Keep running unless the stop signal is set
        try:
            image_data = cam_client.read()  # Read the camera data
            
            if not image_data:  # If no data is returned
                print("No data received from the camera.")
                continue  # Try again

            # Convert the byte data to a numpy array (assuming it's a jpeg or png image)
            np_arr = np.frombuffer(image_data, np.uint8)  # Convert bytes to numpy array
            image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)  # Decode the image
            
            if image is not None:
                # Display the image in a window
                cv2.imshow("Camera Feed", image)
                
                # Save the image to the specified folder
                image_filename = os.path.join(CAMERA_SAVE_PATH, f"image_{image_num}.jpg")
                cv2.imwrite(image_filename, image)  # Save the image
                
                print(f"Image {image_num} saved to {image_filename}")
                image_num += 1  # Increment the image number for the next image
                
                # Check for the 'q' key to quit the display window
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                print("Error: Failed to decode image data")
                
        except Exception as e:
            print(f"Error while reading or processing camera data: {e}")
            break  # Exit the loop if any error occurs
    
    # Clean up and close the window when done
    cv2.destroyAllWindows()
    print("Camera thread terminated.")

# LiDAR thread to handle point cloud data collection and saving
def lidar_thread():
    global lidar_pcd_count
    client = ALSLib.TCPClient.TCPClient(HOST, 8881, 5)  # Assuming LiDAR is on port 8881
    client.connect(5)
    
    while lidar_pcd_count < MAX_LIDAR_PCDS and not stop_threads_event.is_set():  # Stop after 300 PCD files
        # Read LiDAR data
        data = client.read()
        sizeofFloat = 4
        index = 11
        posX, posY, posZ, quatW, quatX, quatY, quatZ, numPoints, timeStart, timeEnd, numberOfBeams = struct.unpack('<fffffffffff', data[0:index * sizeofFloat])
        
        # LiDAR point cloud data
        pointCloudData = data[index * sizeofFloat:]
        point_array = np.frombuffer(pointCloudData, dtype=np.dtype("float32"))
        point_array = np.reshape(point_array, (-1, 4))
        
        # Save the point cloud to file
        pclFileContent = SerializeToPCLFileContent(numPoints, posX, posY, posZ, quatW, quatX, quatY, quatZ, point_array)
        
        filename = get_sensordata_path(f"/pcl/pcl{lidar_pcd_count}.pcd")
        try:
            with open(filename, mode='w') as fileObject:
                fileObject.write(pclFileContent)
            print(f"Point cloud {lidar_pcd_count} saved to {filename}")
            lidar_pcd_count += 1
        except Exception as e:
            print(f"Failed to save point cloud to file: {e}")
        
    stop_threads_event.set()  # Signal the camera thread to stop once LiDAR thread completes

# Main execution flow
if __name__ == "__main__":
    TestContext.client.connect()
    TestContext.client.request_load_scenario("Testing")

    ALSLidar.create_sensor_data_folders()

    sensorprofile_path = ABOAT
    sensor_path_in_file = "Sensors.Sensors.[1]"
    cam_path_in_file = "Sensors.Sensors.[0]"
    #overrides = "{file_path};{sensor_path}.StreamToNetwork;True".format(file_path=sensorprofile_path, sensor_path=sensor_path_in_file)
    #TestContext.client.request_load_situation_layer_with_overrides('DemoSensorsTest', overrides)

    # Start LiDAR thread
    lidar_thread_instance = threading.Thread(target=lidar_thread)
    lidar_thread_instance.start()

    # Start camera thread
    camera_thread_instance = threading.Thread(target=camera_thread)
    camera_thread_instance.start()

    # Wait for the LiDAR thread to finish
    lidar_thread_instance.join()

    # Once the LiDAR thread finishes, the camera thread will also be stopped
    camera_thread_instance.join()
    print("All threads finished.")
