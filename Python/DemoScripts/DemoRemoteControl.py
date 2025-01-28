import time
import threading
import logging
import ALSLib.ALSClient 
import ALSLib.TCPClient

###############################################################################
# Here we show how you can hook into the remote control port of the vehicles 
# and send commands to it. 
# Each vehicle has its own control socket that is defined in the details of 
# the Situation file. 
# Basic vehicles have Throttle, Steering, Brake type of control, but
# optional lists of axes are also available for more complex vehicles. 
# An optional dt command can be used to pause the simulation automatically
#
# This also shows the status message that is recieved by the vehicle
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
	TestContext.client.connect()

	result = TestContext.client.request_load_scenario('OpenOceanNaked')
	result = TestContext.client.request_load_situation('OpenOceanSmallMotorBoat')
		
	remoteControlSocket = ALSLib.TCPClient.TCPClient('127.0.0.1', 7700, 5 )
	remoteControlSocket.connect(5)

	for i in range(0,10):
		throttle = 0.5
		steering = 1.0
		brake = 0.0
		handbrake = 0

		string = 'setControl t:%f s:%f b:%f h:%i' %( throttle, steering, brake, handbrake)
        
		remoteControlSocket.write( string.encode('UTF-8') )
		time.sleep(1)
	for i in range(0,10):
		throttle = 0.5
		steering = -1.0
		brake = 0.0
		handbrake = 0

		string = 'setControl t:%f s:%f b:%f h:%i' %( throttle, steering, brake, handbrake)
        
		remoteControlSocket.write( string.encode('UTF-8') )
		time.sleep(1)

	result = TestContext.client.request_toggle_pause()
	result = TestContext.client.request_destroy_situation()
	result = TestContext.client.request_toggle_pause()



print("Environment RemoteControl ")
RemoteControlScenario()



print("--- end of script ---")
