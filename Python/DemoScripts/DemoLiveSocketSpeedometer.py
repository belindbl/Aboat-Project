import time
import json
import threading
import os
import ALSLib.TCPClient, ALSLib.ALSClient
from matplotlib import pyplot
from ALSLib.ALSHelperFunctionLibrary import get_sensordata_path
from math import pow as cpow, sqrt as csqrt
from PIL import Image
###############################################################################
# This demo shows how to collect the Speedometer data
#
# We use a premade SensorProfile where all the sensors are pre-configured, but
# disabled so we use overrides to only enable the sensor we want in each case
#
# All ports and IDs are defined this way in the particular Sensorprofile
# and can be changed in other use cases
#
###############################################################################

HOST = '127.0.0.1'
savepath = get_sensordata_path('/Speedometer')
vector_output = True

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
	pyplot.savefig(path, format = 'png')
	return path

def vector3length(v):
	v = list(map(lambda x: float(x), v))
	return csqrt(cpow(v[0], 2) + cpow(v[1], 2) + cpow(v[2], 2))

class Speedometer_Data:
	def __init__(self, Time, VTP, VTO, VTS,STP,STO,STS,SV,SAV):
		self.Time = Time
		self.VTP = VTP
		self.VTO = VTO
		self.VTS = VTS
		self.STP = STP
		self.STO = STO
		self.STS = STS
		self.SV = SV
		self.SAV = SAV

	@staticmethod
	def msg_to_datapoint(json_message:dict):
		Time = json_message['Time']
		VTP = json_message['VehicleTransformPosition']
		VTO = json_message['VehicleTransformOrientation']
		VTS = json_message['VehicleTransformScale']
		STP = json_message['SensorTransformPosition']
		STO = json_message['SensorTransformOrientation']
		STS = json_message['SensorTransformScale']
		SV = json_message['SensorVelocity']
		SAV = json_message['SensorAngularVelocity']
		return Speedometer_Data(Time, VTP, VTO, VTS, STP, STO, STS, SV, SAV)

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
	sensor_path_in_file = "Sensors.Sensors.[7]"
	overrides = "{file_path};{sensor_path}.StreamToNetwork;True".format(file_path=sensorprofile_path, sensor_path=sensor_path_in_file)
	TestContext.client.request_load_situation_layer_with_overrides('DemoSensors', overrides)

	port = 8887
	client = ALSLib.TCPClient.TCPClient(HOST, port, 5 )
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

		datas = data.splitlines()
		for d in datas:
			json_data = json.loads(d)
			reads.append(Speedometer_Data.msg_to_datapoint(json_message=json_data))

	if(client.connected()):
		client.disconnect()


	# Use data here
	times = list(map(lambda x: float(x.Time), reads))

	if(vector_output):
		#Map data
		l = list(map(lambda x: vector3length([x.SV[0], x.SV[1], x.SV[2]])/100, reads))

		#Plot data
		file_path = plot_simple_graph('Time (s)','Velocity (M/S)', times, l, 'velocity_over_time.png')
		img = Image.open(file_path)
		img.show()
	else:
		#Map data
		l = list(map(lambda x: x.SV/100, reads))

		#Plot data
		file_path = plot_simple_graph('Time (s)','Velocity (M/S)', times, l, 'velocity_over_time.png')
		img = Image.open(file_path)
		img.show()


