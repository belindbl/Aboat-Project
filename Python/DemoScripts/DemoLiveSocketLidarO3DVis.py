import threading
import numpy as np
import ALSLib.TCPClient, ALSLib.ALSClient 
import ALSLib.ALSPointCloudVisualisation as ALSLidarVis
import ALSLib.ALSHelperLidarLibrary as ALSLidar
from ALSLib.ALSHelperFunctionLibrary import get_sensordata_path

###############################################################################
# This demo shows how to collect the Lidar data and visualize
# them with a library using Open3D which requires Python 3.9
#
# We use a premade SensorProfile where all the sensors are pre-configured, but 
# disabled so we use overrides to only enable the sensor we want in each case
###############################################################################

HOST = '127.0.0.1'

def myMessageHandler(rawMessage):
	str = rawMessage.decode('utf-8')
	#print ('\n---> recieved Raw message: '+str)
	
	cmdList = str.split(" ")
	if cmdList[0].startswith("EndCondition") :
		TestContext.client.request_destroy_situation()

		TestContext.lock.acquire()
		TestContext.testEnded = True
		print("setting TestEnded")
		TestContext.lock.release()
		

class TestContext:
	lock = threading.Lock()
	testEnded = False
	simulation_control_port = 9000
	client = ALSLib.ALSClient.Client((HOST, simulation_control_port),myMessageHandler)
	

if __name__ == "__main__":
	TestContext.client.connect()
	TestContext.client.request_load_scenario('Default_Scenario')

	ALSLidar.create_sensor_data_folders()

	sensorprofile_path = "EgoVehicleSettings\DemoAllSensors.SensorProfile.ini"
	sensor_path_in_file = "Sensors.Sensors.[1]"
	overrides = "{file_path};{sensor_path}.StreamToNetwork;True".format(file_path=sensorprofile_path, sensor_path=sensor_path_in_file)
	TestContext.client.request_load_situation_layer_with_overrides('DemoSensors', overrides)

	lazer_proximity_port = 8881
	client = ALSLib.TCPClient.TCPClient(HOST, lazer_proximity_port, 5 )
	client.connect(5)

	imageNum = 0
	
	vis = ALSLidarVis.ALSPointCloudVisualiser()
	

	while(True):
		data = client.read()
		posX,posY,posZ,quatW,quatX,quatY,quatZ,numPoints,timeStart,timeEnd,\
    	numberOfBeams, pointCloudData = ALSLidar.get_lidar_header(data)

		point_array = np.frombuffer(pointCloudData, dtype=np.dtype("float32"))
		point_array = np.reshape(point_array, (-1,4))

		vis.update_geometry(ALSLidarVis.get_o3d_points(point_array))
			

		# This code saves the point cloud to file that can be opened with cloud compare
		pclFileContent = ALSLidar.SerializeToPCLFileContent(numPoints, posX, posY, posZ, quatW, quatX, quatY, quatZ, point_array)
		filename = get_sensordata_path('/pcl/pcl'+ str(imageNum)+'.pcd')
		fileObject = open(filename, mode='w')
		fileObject.write(pclFileContent)

		




