import threading, struct, math
import numpy as np
import matplotlib.pyplot as plt
import ALSLib.TCPClient, ALSLib.ALSClient 
import ALSLib.ALSHelperLidarLibrary as ALSLidar
from ALSLib.ALSHelperFunctionLibrary import get_sensordata_path

###############################################################################
# This demo shows how to collect the Lidar data
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

def SerializeToPCLFileContent(numPoints, posX, posY, posZ, quatW, quatX, quatY, quatZ, point_array):
	pclFileContent = '# .PCD v.7 - Point Cloud Data file format\nVERSION .7\nFIELDS x y z rgb\n\
	SIZE 4 4 4 4\nTYPE F F F U\nCOUNT 1 1 1 1\nWIDTH %d\nHEIGHT 1\nVIEWPOINT %f %f %f %f %f %f %f \n\
	POINTS %d \nDATA ascii\n' % (int(numPoints), posX, posY, posZ, quatW, quatX, quatY, quatZ, int(numPoints))
	for p in point_array :
		intensity = 1000
		if not math.isinf(p[3]):
			intensity = int(p[3])
		pclFileContent +=  '%.5f %.5f %.5f %d\n' % (p[0], p[1], p[2], intensity) 
	return pclFileContent

fig = plt.figure()
def plot_ply(points):
	ax = fig.add_subplot(111, projection='3d')
	x,y,z = points[:,0],points[:,1], points[:,2]
	ax.scatter(x, y, z, c='r', marker='o')
	ax.set_xlabel('X Label')
	ax.set_ylabel('Y Label')
	ax.set_zlabel('Z Label')
	plt.pause(0.07)
	plt.clf()
	#plt.show()	

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
	while(True):
		data = client.read()
		sizeofFloat = 4
		index = 11
		posX,posY,posZ,quatW,quatX,quatY,quatZ,numPoints,timeStart,timeEnd,\
		numberOfBeams = struct.unpack('<fffffffffff', data[0:index*sizeofFloat])
		print('PCL: %f %f %f',posX, posY, posZ)
		
		#this code does the plot
		pointCloudData = data[index*sizeofFloat:]
		point_array = np.frombuffer(pointCloudData, dtype=np.dtype("float32"))
		point_array = np.reshape(point_array, (-1,4))
		plot_ply(point_array)

		# this code saves the point cloud to file that can be opened with cloud compare
		pclFileContent = SerializeToPCLFileContent(numPoints, posX, posY, posZ, quatW, quatX, quatY, quatZ, point_array)
		filename = get_sensordata_path('/pcl/pcl'+ str(imageNum)+'.pcd')
		fileObject = open(filename, mode='w')
		fileObject.write(pclFileContent)




