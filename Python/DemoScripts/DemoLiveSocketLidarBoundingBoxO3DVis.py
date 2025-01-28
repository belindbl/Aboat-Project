import threading, struct, json, time
import numpy as np
import ALSLib.TCPClient, ALSLib.ALSClient 
import ALSLib.ALSPointCloudVisualisation as ALSLidarVis
import ALSLib.ALSHelperLidarLibrary as ALSLidar
import ALSLib.ALSHelperFunctionLibrary as ALSFunc


###############################################################################
# In this demo we receive a point cloud from a lidar and 3D bounding boxes of 
# boat objects from FilteredObjectGetter. These are then visualized simultaneously
# with Open3D based functions from ALSPointCloudVisualisation
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
	# TestContext.client.request_load_scenario('Default_Scenario')

	TestContext.client.request_load_situation('TestBoundingBox3D')

	sensor_group_port = 8880
	client = ALSLib.TCPClient.TCPClient(HOST, sensor_group_port, 5 )
	client.connect(5)
	

	imageNum = 0
	
	vis = ALSLidarVis.ALSPointCloudVisualiser()
	

	while(True):
		data = client.read()

		sizeofFloat = 4
		index = 0
		point_array = []
		group_sensor_amount, index = ALSFunc.ReadUint32(data, index)


		for i in range(group_sensor_amount):
			sensor_type, index 		= ALSFunc.ReadString(data, index)
			sensor_path, index 		= ALSFunc.ReadString(data, index)
			print("sensor ", i,": Type ", sensor_type, "  path:", sensor_path)

			if sensor_type == "LID":
				
				LidarHeaderSize = 11
				LidarDataSize = 4

				posX,posY,posZ,quatW,quatX,quatY,quatZ,numPoints,timeStart,timeEnd,\
				numberOfBeams, pointCloudData, index = ALSLidar.get_lidar_header_from_group(data, index)

				for i in range(int(numPoints)):
					X,Y,Z,W = struct.unpack('<ffff', data[index:index + (LidarDataSize*sizeofFloat)])
					index += LidarDataSize*sizeofFloat
					point_array.append([X, Y, Z, W])

			if sensor_type == "FOG":
				fog, index = ALSFunc.ReadString(data, index)
				print(fog)
				parsed_string 		= json.loads(fog)
				time 				= float(parsed_string['T'])
				tracked_objects =  parsed_string['MSG']
				for a_obj in tracked_objects:
					if None is not a_obj.get('BoundingBox'):
						BoundingBox			= a_obj['BoundingBox']
						points = [[float(x), float(y), float(z)] for x, y, z in BoundingBox]
						vis.add_box_from_points(points)
				vis.spawn_in_boxes()
		
		point_array = np.reshape(point_array, (-1,4))
		vis.update_geometry(ALSLidarVis.get_o3d_points(point_array))


		
			

		




