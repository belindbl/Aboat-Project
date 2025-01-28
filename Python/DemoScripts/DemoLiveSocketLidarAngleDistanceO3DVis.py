import threading, json
import numpy as np
import ALSLib.TCPClient, ALSLib.ALSClient 
import ALSLib.ALSPointCloudVisualisation as ALSLidarVis
import ALSLib.ALSHelperLidarLibrary as ALSLidar
from ALSLib.ALSHelperFunctionLibrary import get_sensordata_path

###############################################################################
# This demo shows how to collect the Lidar data with angle and distance and 
# visualize them with a library using Open3D which requires Python 3.9
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
	overrides  = f"{sensorprofile_path};{sensor_path_in_file}.StreamToNetwork;True<br>"
	overrides += f"{sensorprofile_path};{sensor_path_in_file}.SensorConfigFileName;ScannerAngleDistance"
	TestContext.client.request_load_situation_layer_with_overrides('DemoSensors', overrides)

	response = TestContext.client.request_get_lidar_info('Ego', 'Sensors.[1]')

	print(response)
	response = json.loads(response)

	lazer_proximity_port = 8881
	client = ALSLib.TCPClient.TCPClient(HOST, lazer_proximity_port, 5 )
	client.connect(5)

	imageNum = 0
	num_points_in_frame = 0
	current_frame = -1


	frame_builder = ALSLidarVis.FrameBuilder()

	while(True):
		data = client.read()
		posX,posY,posZ,quatW,quatX,quatY,quatZ,num_cols,num_beams_per_col,col_time,FrameID,\
		ColId, pointcloud_data = ALSLidar.get_lidar_header_angle_distance(data)

		if current_frame == -1: 
			current_frame = FrameID

		#This code does the convertion
		readings_array = np.frombuffer(pointcloud_data, dtype=np.dtype("float32"))

		frame_builder.add_column_points(num_beams_per_col, ColId, readings_array, response)
	
		if FrameID != current_frame :
			print(f'PCL: {posX} {posY} {posZ}  Frame ID: {FrameID} - num col: {num_cols}')
			#We can send the previous frame
			pclFileContent = ALSLidar.SerializeToPCLFileContent(num_points_in_frame, posX, posY, posZ, quatW, quatX, quatY, quatZ, frame_builder.point_array)
			filename = get_sensordata_path('/pcl/pcl'+ str(imageNum)+'.pcd')
			fileObject = open(filename, mode='w')
			fileObject.write(pclFileContent)

			frame_builder.display_frame()

			#And prepare for the next one
			current_frame = FrameID
