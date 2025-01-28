import threading, struct
import numpy as np
import ALSLib.TCPClient, ALSLib.ALSClient

###############################################################################
# This demo shows how to collect the Radar data
#
# We use a premade SensorProfile where all the sensors are pre-configured, but 
# disabled so we use overrides to only enable the sensor we want in each case
###############################################################################

HOST = '127.0.0.1'

class radar_datapoint():
	def __init__(self, velocity, azimuth, altitude, depth):
		self.velocity = velocity
		self.azimuth = azimuth
		self.altitude = altitude
		self.depth = depth

	@staticmethod    
	def msg_to_datapoints(msg):
		data_points = []
		point_amount = struct.unpack('<i', msg[0:4])[0] *4
		time = struct.unpack('f', msg[-4:])[0]
		#print("Reading {} point from message".format(point_amount))
		data = msg[4:-4]
		array = np.frombuffer(data, dtype=np.dtype("float32"))
		array = np.reshape(array, (-1,4))
		for point in array:
			dp = radar_datapoint(point[0],point[1],point[2],point[3])
			# print("Velocity{}, Azimuth{}, Altitude{}, Depth{}".format(point[0],point[1],point[2],point[3]))
			data_points.append(dp)
		return data_points, time

	def serialize(self):
		str = "Velocity{}, Azimuth{}, Altitude{}, Depth{}".format(self.velocity,self.azimuth,self.altitude,self.depth)
		return str

# Write your own message handler to decide how to react to collisions, triggers etc..
def myMessageHandler(rawMessage):
	str = rawMessage.decode('utf-8')	
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
	sensorprofile_path = "EgoVehicleSettings\DemoAllSensors.SensorProfile.ini"
	sensor_path_in_file = "Sensors.Sensors.[2]"
	overrides = "{file_path};{sensor_path}.StreamToNetwork;True".format(file_path=sensorprofile_path, sensor_path=sensor_path_in_file)
	TestContext.client.request_load_situation_layer_with_overrides('DemoSensors', overrides)

	Stereo_camera_port = 8882
	print( "Connecting sensor socket to  " + HOST + " " + str(Stereo_camera_port) )
	client = ALSLib.TCPClient.TCPClient(HOST, Stereo_camera_port, 5 )
	client.connect(5)

	frame = 0
	while(frame < 100):
		try:
			data = client.read()
			readings, time = radar_datapoint.msg_to_datapoints(data)
			print(time)
			print("Received ", len(readings)," this frame")
			frame += 1
		except ALSLib.TCPClient.TCPConnectionError:
			print('Connection closed')
			break

	#disconnect
	if(client.connected()):
		client.disconnect()
