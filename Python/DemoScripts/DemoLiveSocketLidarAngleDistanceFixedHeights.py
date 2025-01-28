import threading, struct, math, json
import numpy as np
import matplotlib.pyplot as plt
import ALSLib.TCPClient, ALSLib.ALSClient
import ALSLib.ALSHelperLidarLibrary as ALSLidar
from ALSLib.ALSHelperFunctionLibrary import get_sensordata_path

###############################################################################
# This demo shows how to collect the Lidar data with angle and distance
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
	plt.pause(0.02)
	plt.clf()


# r is the distance of the reading
# theta is angle between Z and r  (!!not (x.y) and r)
# phi is the angle in the plane (xy)
def reading_to_3d(r,theta,phi, intensity):
	theta_r = np.deg2rad(theta)
	phi_r = np.deg2rad(phi)
	x = r * math.sin(theta_r) * math.cos(phi_r)
	y = r * math.sin(theta_r) * math.sin(phi_r)
	z = r * math.cos(theta_r)
	return [x,y,z,intensity]

if __name__ == "__main__":
	TestContext.client.connect()
	TestContext.client.request_load_scenario('Default_Scenario')
	
	ALSLidar.create_sensor_data_folders()

	sensorprofile_path = "EgoVehicleSettings\DemoAllSensors.SensorProfile.ini"
	sensor_path_in_file = "Sensors.Sensors.[1]"
	overrides  = f"{sensorprofile_path};{sensor_path_in_file}.StreamToNetwork;True<br>"
	overrides += f"{sensorprofile_path};{sensor_path_in_file}.SensorConfigFileName;ScannerAngleDistanceFixedHeights"
	TestContext.client.request_load_situation_layer_with_overrides('DemoSensors', overrides)

	response = TestContext.client.request_get_lidar_info('Ego', 'Sensors.[1]')
	# print(response)
	response = json.loads(response)
	print(response)

	lazer_proximity_port = 8881
	client = ALSLib.TCPClient.TCPClient(HOST, lazer_proximity_port, 5 )
	client.connect(5)

	imageNum = 0
	num_points_in_frame = 0
	current_frame = -1


	point_array = []
	while(True):
		data = client.read()
		sizeofFloat, sizeofInt = 4, 4
		index = 8*sizeofFloat+4*sizeofInt
		posX,posY,posZ,quatW,quatX,quatY,quatZ,num_cols,num_beams_per_col,col_time,FrameID,\
		ColId = struct.unpack('<fffffffiifii', data[0:index])
		print(f'PCL: {posX} {posY} {posZ}  Frame ID: {FrameID} - num col: {num_cols}')

		if current_frame == -1: 
			current_frame = FrameID

		#this code does the convertion
		pointcloud_data = data[index:]
		readings_array = np.frombuffer(pointcloud_data, dtype=np.dtype("float32"))
		assert(num_beams_per_col == len(response['beam_altitude_angles']) )
		h_steps = response['span_width_angle_degree']/response['span_width_steps'] 
		for count in range(num_beams_per_col):
			d = readings_array[count]
			angle = response['beam_altitude_angles'][count]
			intensity = readings_array[count+num_beams_per_col]

			reading = reading_to_3d(d, 90-angle, 180-h_steps*ColId, intensity )
			# reading = reading_to_3d(d, 90-(count/v_span_degree-v_span_degree*0.5), 180-h_steps*ColId, readings_array[count+num_beams_per_col] )
			show_real_position = False
			if show_real_position:
				reading = [reading[0]+posX, reading[1]+posY, reading[2]+posZ, reading[3]]
			else:
				reading = [reading[0], reading[1], reading[2], reading[3]]
			point_array.append(reading)
			num_points_in_frame += 1
			if count>=num_beams_per_col-1: # after the list of distances we get the list of intensities
				break

		if FrameID != current_frame :
			#we can send the previous frame
			pclFileContent = SerializeToPCLFileContent(num_points_in_frame, posX, posY, posZ, quatW, quatX, quatY, quatZ, point_array)
			filename = get_sensordata_path('/pcl/pcl'+ str(imageNum)+'.pcd')
			fileObject = open(filename, mode='w')
			fileObject.write(pclFileContent)
			point_array = np.reshape(point_array, (-1,4))
			plot_ply(point_array)


			# and prepare for the next one
			current_frame = FrameID
			point_array = []
			num_points_in_frame = 0
