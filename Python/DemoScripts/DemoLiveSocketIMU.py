import time
import json
import threading, re, os
from matplotlib import pyplot
from math import pow, sqrt
import ALSLib.TCPClient, ALSLib.ALSClient
from ALSLib.ALSHelperFunctionLibrary import get_sensordata_path

###############################################################################
# This demo shows how to collect the IMU data
#
# We use a premade SensorProfile where all the sensors are pre-configured, but
# disabled so we use overrides to only enable the sensor we want in each case
#
# All ports and IDs are defined this way in the particular Sensorprofile
# and can be changed in other use cases
#
# To use JSON output, add <_DataAsJSON>True</_DataAsJSON> to the IMU sensor config
# and comment out lines 125-126 to read the JSON data
#
###############################################################################

HOST = '127.0.0.1'

samples_in_step = 200
sample_time_step = 0.5

savepath = get_sensordata_path('/IMU')

class IMU_datapoint():
	def __init__(self, time, orientation, angular_velocity, linear_acceleration):
		self.time = time
		self.orientation = orientation
		self.linear_acceleration = linear_acceleration
		self.angular_velocity = angular_velocity

	@staticmethod
	def msg_to_datapoint(msg):
		pattern = r'"T":(-?\d*.?\d*)."O":\((-?\d*.?\d*,-?\d*.?\d*,-?\d*.?\d*,-?\d*.?\d*)\),"AV":\((-?\d*.?\d*,-?\d*.?\d*,-?\d*.?\d*)\),"LA":\((-?\d*.?\d*,-?\d*.?\d*,-?\d*.?\d*)\)'
		p = re.compile(pattern)
		match = p.search(msg)

		return IMU_datapoint(match[1], match[2], match[3], match[4])

	@staticmethod
	def msg_to_datapoint_json(msg):
		pattern = r'"T":(-?\d*.?\d*)."O":\[(-?\d*.?\d*,-?\d*.?\d*,-?\d*.?\d*,-?\d*.?\d*)\],"AV":\[(-?\d*.?\d*,-?\d*.?\d*,-?\d*.?\d*)\],"LA":\[(-?\d*.?\d*,-?\d*.?\d*,-?\d*.?\d*)\]'
		p = re.compile(pattern)
		match = p.search(msg)

		return IMU_datapoint(match[1], match[2], match[3], match[4])

def vector3length(v):
	v = list(map(lambda x: float(x), v))
	return sqrt(pow(v[0], 2) + pow(v[1], 2) + pow(v[2], 2))

def plot_simple_graph(xlabel,ylabel,xdata,ydata,filename):
	pyplot.clf()
	if not os.path.exists(savepath):
		os.makedirs(savepath)

	#Create a unique filepath
	c = 0
	path = os.path.join(savepath, filename)
	p = os.path.abspath(path)

	while os.path.exists(p):
		splits = os.path.splitext(filename)
		newname = splits[0] + str(c) + splits[1]
		path = os.path.join(savepath, newname)
		p = os.path.abspath(path)
		c += 1

	#Create plot
	pyplot.xlabel(xlabel)
	pyplot.ylabel(ylabel)
	pyplot.plot(xdata, ydata)

	#Save plot
	pyplot.show()
	pyplot.savefig(path, format = 'png')


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
	sensor_path_in_file = "Sensors.Sensors.[3]"
	overrides = "{file_path};{sensor_path}.StreamToNetwork;True".format(file_path=sensorprofile_path, sensor_path=sensor_path_in_file)
	TestContext.client.request_load_situation_layer_with_overrides('DemoSensors', overrides)

	imu_port = 8883
	client = ALSLib.TCPClient.TCPClient(HOST, imu_port, 5 )
	client.connect(5)

	reads = []
	collection_duration = 10
	start_time = time.time()
	run_time = 0
	while run_time < collection_duration:
		run_time = time.time() - start_time
		print("Collecting the data {}\r".format(run_time), end="", flush=True)
		data = client.read()
		data = data.decode('utf8')

		# reads.append(IMU_datapoint.msg_to_datapoint(data))
		datas = data.splitlines()
		for d in datas:
			json_data = json.loads(d)
			reads.append(IMU_datapoint.msg_to_datapoint_json(d))

	print("Done generating the graph")
	if(client.connected()):
		client.disconnect()

	times = list(map(lambda x: float(x.time), reads))
	las = map(lambda x: x.linear_acceleration, reads)
	las = list(map(lambda x: vector3length(x.split(',')), las))
	avs = list(map(lambda x: x.angular_velocity, reads))
	avs = list(map(lambda x: vector3length(x.split(',')), avs))
	plot_simple_graph('Time (s)','Linear acceleration (M/S²)', times, las, 'linear_acceleration_over_time.png')
	plot_simple_graph('Time (s)','Angular velocity (°/s)', times, avs, 'angular_velocity_over_time.png')
