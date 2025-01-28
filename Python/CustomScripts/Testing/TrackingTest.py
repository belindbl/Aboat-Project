import threading, time, json
import ALSLib.TCPClient, ALSLib.ALSClient
import ALSLib.ALSHelperFunctionLibrary as ALSFunc
import ALSLib.ALSHelperImageLibrary as ALSImg
import ALSLib.ALSHelperLidarLibrary as ALSLidar
import torch
import cv2
import math

# Implement LIDAR visualisation alongside the YOLO output window
# YOLO working 2025-01-15 13:50
# Possibly evaluate a later model? Both v7 and v9 have an MIT-licensed rewrite
# https://github.com/WongKinYiu/YOLO

HOST = '127.0.0.1'

ALSweights = "E:/Team1Docs/yolov5/weights_ailivesim/best.pt"
MixedWeights = "E:/Team1Docs/yolov5/weights_mixed/best.pt"

# Initialize YOLOv5 model
print("Loading YOLOv5 model...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torch.hub.load('ultralytics/yolov5', 'custom',path=ALSweights)
model.to(device)
model.conf = 0.25  # Confidence threshold
model.iou = 0.45   # NMS IOU threshold


# Which library has GetObjectLocation?
"""
def objectDistance(object1, object2):
    x1 = GetObjectLocation(object1).[0]
    y1 = GetObjectLocation(object1).[1]
    z1 = GetObjectLocation(object1).[2]

    x2 = GetObjectLocation(object2).[0]
    y2 = GetObjectLocation(object2).[0]
    z2 = GetObjectLocation(object2).[0]
    distance = math.sqrt((x1-x2)^2+(y1-y2)^2+(z1-z2)^2)
    return distance
"""

# Message handler
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

# Returns processed image, bbox, label and confidence
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

    # Retrieve frame bounding boxes, labels and confidences
    bbox = results.xyxy[0].cpu().numpy() #bbox in xyxy format
    labels = results.names
    conf = results.xyxy[0][:, 4].cpu().numpy() # Confidence scores
    return processed_image, bbox, labels, conf

"""def SerializeToPCLFileContent(numPoints, posX, posY, posZ, quatW, quatX, quatY, quatZ, point_array, timeEnd):

    posY = -posY # Flip coordinates over y-axis to avoid mirrored display in o3d
    pclFileContent = '# .PCD v.7 - Point Cloud Data file format\nVERSION .7\nFIELDS x y z rgb time\n'\
    'SIZE 4 4 4 4 4\nTYPE F F F U F\nCOUNT 1 1 1 1 1\nWIDTH %d\nHEIGHT 1\nVIEWPOINT %f %f %f %f %f %f %f\n'\
    'POINTS %d\nDATA ascii\n' % (int(numPoints), posX, posY, posZ, quatW, quatX, quatY, quatZ, int(numPoints))
    
    for p in point_array:
        intensity = 1000
        if not math.isinf(p[3]):
            intensity = int(p[3])
        pclFileContent += '%.5f %.5f %.5f %d %.5f\n' % (p[0], p[1], p[2], intensity, timeEnd) 
    return pclFileContent

def process_pcloud(pcd):
    

"""

if __name__ == "__main__":
    TARGET_FPS = 10  # Set your desired frame rate
    FRAME_TIME = 1/TARGET_FPS  # Time per frame in seconds
    
    # Initialize simulation
    TestContext.client.connect()
    TestContext.client.request_load_scenario('Default_Scenario')
    overrides = "EgoVehicleSettings\DemoAllSensors.SensorProfile.ini;Sensors.Sensors.[0].StreamToNetwork;True"
    TestContext.client.request_load_situation_layer_with_overrides('DemoSensorsTest', overrides)
    
    # Get the sensor list
    time.sleep(1)
    sensorlist = TestContext.client.get_sensor_list()
    parsed_json = json.loads(sensorlist)
    print("\n")
    print(parsed_json['sensors'][10])
    

	# Find the host and port of the camera
    for x in parsed_json['sensors']:
        if x['path'] == 'Sensors.[0]':
            sensor_cam = x
            break
    camera_port = sensor_cam['sensor_port']
    host = sensor_cam['sensor_ip']

    """
    # Find the host and port of the LiDAR
    for y in parsed_json['sensors']:
        if y['path'] == 'Sensors.[1]':
            sensor_lidar = y
            break
    lidar_port = sensor_lidar['sensor_port']
    host = sensor_lidar['sensor_ip']
    """
	#The camera port is defined inside the situation ini files, can be changed in Sensor Editor. 
    print( "Connecting sensor socket to  " + host + " " + str(camera_port) )
    client = ALSLib.TCPClient.TCPClient(host, camera_port, 5 ) # Last arg is timeout in seconds
    client.connect(5)

    imageNum = 0
    while(imageNum < 200):
        frame_start_time = time.time()  # Start timing the frame

        # Read data from stream
        data = client.read()
        index = 0
        img, index, width, height = ALSFunc.ReadImage_Stream(data, index)

        processed_img = process_frame(img)
        print(processed_img)

        ALSImg.JustDisplay(processed_img)
        
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