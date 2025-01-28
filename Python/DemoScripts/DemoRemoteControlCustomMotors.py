import numpy as np
import PIL.Image
from io import BytesIO
import ALSLib.ALSClient 
import time
import threading
import logging
import ALSLib.TCPClient
from ast import literal_eval

###############################################################################
# Here we show how you can hook into the remote control port of the vehicles 
# and send commands to it. 
# Each vehicle has its own control socket that is defined in the details of 
# the Situation file. 
# Basic vehicles have Throttle, Steering, Brake type of control, but
# optional lists of axes are also available for more complex vehicles. 
# An optional dt command can be used to pause the simulation automatically
#
# This also shows the status message that is received by the vehicle
###############################################################################


_L = logging.getLogger(__name__)
_L.setLevel(0)

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
	if cmdList[0].startswith("EndCondition") :
		TestContext.client.request_destroy_situation()

		TestContext.lock.acquire()
		TestContext.testEnded = True
		print("setting TestEnded")
		TestContext.lock.release()

	if cmdList[0].startswith("Status") :
		TestContext.lock.acquire()
		TestContext.vehicleStatus.SimulationTime = float(cmdList[1])
		TestContext.vehicleStatus.StatusString = cmdList[2]
		print('Received ', len(cmdList[2]),' Status:', TestContext.vehicleStatus.StatusString)
		TestContext.lock.release()
	

class TestContext:
	lock = threading.Lock()
	testEnded = False
	echo_port = 9000
	localhost = 'localhost'
	client = ALSLib.ALSClient.Client((localhost, echo_port),myMessageHandler)
	PORT = 7700
	HOST = '127.0.0.1'
	vehicleStatus = VehicleStatus()

def RemoteControlScenario():
	print("--- start of script ---")
	TestContext.client.connect()

	result = TestContext.client.request_load_scenario('OpenOceanNaked')
	result = TestContext.client.request_load_situation('DemoABoat')
		
	remoteControlSocket = ALSLib.TCPClient.TCPClient('127.0.0.1', 7700, 5 )
	remoteControlSocket.connect(5)

	throttle = 0.0
	steering = 0.0
	brake = 0.0
	handbrake = 0

	for i in range(0,10):
		# a0 to a31
		string = 'setControl t:%f s:%f b:%f h:%i a0:%f a1:%f a2:%f a3:%f a4:%f a5:%f a6:%f a7:%f' %( throttle, steering, brake, handbrake, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
		print("right")
		remoteControlSocket.write( string.encode('UTF-8') )

	time.sleep(10)

	for i in range(0,10):

		string = 'setControl t:%f s:%f b:%f h:%i a0:%f a1:%f a2:%f a3:%f a4:%f a5:%f a6:%f a7:%f' %( throttle, steering, brake, handbrake, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0)
		print("left")
		remoteControlSocket.write( string.encode('UTF-8') )
	time.sleep(10)

	string = 'setControl t:%f s:%f b:%f h:%i a0:%f a1:%f a2:%f a3:%f a4:%f a5:%f a6:%f a7:%f' %( throttle, steering, brake, handbrake, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
	print("stop")
	remoteControlSocket.write( string.encode('UTF-8') )



print("Environment RemoteControl ")
RemoteControlScenario()

print("--- end of script ---")
