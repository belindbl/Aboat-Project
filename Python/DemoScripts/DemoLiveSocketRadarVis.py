import json, time, random
import numpy as np
import threading
import ALSLib.TCPClient, ALSLib.ALSClient
import ALSLib.ALSRadarVis
from multiprocessing import Process, Queue as MPQueue
from queue import Empty as QueueEmpty

###############################################################################
# This demo shows how to collect the Radar data and visualize it
#
# We use a premade SensorProfile where all the sensors are pre-configured, but 
# disabled so we use overrides to only enable the sensor we want in each case
###############################################################################

HOST = '127.0.0.1'

class VehicleStatus:
	def __init__(self):
		self.SimulationTime = 0.0
		self.StatusString = ''
	def Copy(self, other):
		self.SimulationTime = other.SimulationTime
		self.StatusString = other.StatusString


def myMessageHandler(rawMessage):
	str = rawMessage.decode('utf-8')
	cmdList = str.split(" ")
	
	if len(cmdList) <= 1 and cmdList[0].startswith("EndCondition") or cmdList[1].startswith("Collision"):
		print("Collision detected: -> setting TestEnded")
		TestContext.client.request_destroy_situation()
		TestContext.lock.acquire()
		TestContext.testEnded = True
		TestContext.lock.release()

	if cmdList[0].startswith("Status"):
		TestContext.lock.acquire()
		TestContext.vehicleStatus.SimulationTime = float(cmdList[1])
		if(len(cmdList[2]) > 0):
			TestContext.vehicleStatus.StatusString = cmdList[2]
		TestContext.lock.release()


class TestContext:
	lock = threading.Lock()
	testEnded = False	
	simulation_control_port = 9000
	client = ALSLib.ALSClient.Client((HOST, simulation_control_port),myMessageHandler)
	vehicleStatus = VehicleStatus()

def main_update(data_queue:MPQueue, stop_queue:MPQueue):
	print( "Connecting sensor socket to  " + HOST + " " + str(8882) )
	sensor_client = ALSLib.TCPClient.TCPClient(HOST, 8882, 5 )
	sensor_client.connect(5)
	while True:
		try:
			data = sensor_client.read()
			data_json = json.loads(data)
			for reading in data_json.get("data"):
				angle = reading[1]
				distance = reading[2] / 100.0
				x = np.sin(angle) * distance
				y = np.cos(angle) * distance
				dot = ALSLib.ALSRadarVis.Dot(x, y, 3, 3)
				dot.set_rgba(1, 1, 0, .1)
				data_queue.put(dot)
			
		except ALSLib.TCPClient.TCPConnectionError:
			print('Connection closed')

		try:
			stop_message = stop_queue.get_nowait()
			if stop_message:
				break
		except QueueEmpty:
			pass

if __name__ == "__main__":

	TestContext.client.connect()
	TestContext.client.request_load_scenario('Default_Scenario')
	sensorprofile_path = "EgoVehicleSettings\RadarVisualizationDemo.SensorProfile.ini"
	overrides = "{file_path};Sensors.Sensors.[0].StreamToNetwork;True".format(file_path=sensorprofile_path)
	TestContext.client.request_load_situation_layer_with_overrides('DemoRadarVisualization', overrides)

	data_queue = MPQueue()
	stop_queue = MPQueue()
	radar_visualization = ALSLib.ALSRadarVis.RadarVisualization(data_queue, "ALSRadarVisualisationConfig.json")
	print("Creating data thread")
	data_process = Process(target=main_update, args=(data_queue,stop_queue,))
	data_process.start()

	radar_visualization.draw_range_circles(True)
	radar_visualization.run()

	stop_queue.put(True)
	print("Stopped data thread")
	