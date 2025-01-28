import threading, struct, math
import time
import numpy as np
import matplotlib.pyplot as plt
import ALSLib.TCPClient, ALSLib.ALSClient 
import ALSLib.ALSHelperFunctionLibrary as ALSFunc
import ALSLib.ALSHelperImageLibrary as ALSImg
from ALSLib.ALSHelperFunctionLibrary import get_sensordata_path

###############################################################################
# This demo shows how to operate a sensor group
###############################################################################

HOST = 'localhost'

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

	TestContext.client.request_load_situation('SensorGroupTest')

	sensor_group_port = 8881
	client = ALSLib.TCPClient.TCPClient(HOST, sensor_group_port, 5 )

	client.connect(5)
	imageNum = 0
	while(True):
		data = client.read()
		print('len: ' + str(len(data)))
		sizeofFloat = 4

		index = 0
		group_sensor_amount, index = ALSFunc.ReadUint32(data, index)
		print("sensor count:", str(group_sensor_amount))

		PCpoints = []
		received_images = []

		for i in range(group_sensor_amount):
			sensor_type, index 		= ALSFunc.ReadString(data, index)
			sensor_path, index 		= ALSFunc.ReadString(data, index)
			print("sensor ", i,": Type ", sensor_type, "  path:", sensor_path)

			if sensor_type == "LID":
				
				LidarHeaderSize = 11
				posX,posY,posZ,quatW,quatX,quatY,quatZ,numPoints,timeStart,timeEnd,\
					numberOfBeams = struct.unpack('<fffffffffff', data[index:index+(LidarHeaderSize*sizeofFloat)])
				index += LidarHeaderSize * sizeofFloat
				print('index: ' + str(index))

				LidarDataSize = 4
				for i in range(int(numPoints)):

					X,Y,Z,W = struct.unpack('<ffff', data[index:index + (LidarDataSize*sizeofFloat)])
					index += LidarDataSize*sizeofFloat

					PCpoints.append([X, Y, Z, W])
					print('Point: %.2f %.2f %.2f' % (X, Y, Z))

			if sensor_type == "CAM":

				image, index, image_width, image_height	= ALSFunc.ReadImage_Group(data, index)
				received_images.append(image)

				extra_string, index 	= ALSFunc.ReadString(data, index)
				print("extra string: ", extra_string)

			if sensor_type == "SPE":
				speed, index = ALSFunc.ReadString(data, index)
				print(speed)

			if sensor_type == "FOG":
				asd, index = ALSFunc.ReadString(data, index)
				print(asd)

			if sensor_type == "RAD":
				DataCount, index = ALSFunc.ReadUint32(data, index)
				RadarDataSize = 4
				for i in range(DataCount):
					Altitude,Azimuth,Depth,Velocity = struct.unpack('<ffff', data[index:index + (RadarDataSize*sizeofFloat)])
					index += RadarDataSize*sizeofFloat

					print("%d: %f %f %f %f" % (i, Altitude, Azimuth, Depth, Velocity))

			if sensor_type == "LAS":
				LaserData, index = ALSFunc.ReadString(data, index)
				print(LaserData)

			if sensor_type == "GNS":
				GNSSData, index = ALSFunc.ReadString(data, index)
				print(GNSSData)

			if sensor_type == "SPL":
				SplineData, index = ALSFunc.ReadString(data, index)
				print(SplineData)

			if sensor_type == "IMU":
				IMUData, index = ALSFunc.ReadString(data, index)
				print(IMUData)

			if sensor_type == "OCC":
				OCCData, index = ALSFunc.ReadString(data, index)
				idx = OCCData.index(';')
				
				time = float(OCCData[:idx])
				states = str(OCCData[idx+1:-2])
				array = [state=='1' for state in states]
				print("data: ", str(time), " : ", len(array), "\n", array)

		if len(received_images) > 0:
			stack = ALSImg.StackImages(received_images)
			ALSImg.JustDisplay(stack)

		if len(PCpoints) > 0:
			point_array = np.array(PCpoints)
			plot_ply(point_array)

			# This code saves the point cloud to file that can be opened with cloud compare
			pclFileContent = SerializeToPCLFileContent(numPoints, posX, posY, posZ, quatW, quatX, quatY, quatZ, point_array)
			filename = get_sensordata_path('/pcl/pcl'+ str(imageNum)+'.pcd')
			fileObject = open(filename, mode='w')
			fileObject.write(pclFileContent)
			imageNum = imageNum + 1
