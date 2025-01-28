import ALSLib.ALSClient 
import time
import threading
import logging
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

def DemoVelocity():
	SettingsPath = "EgoVehicleSettings\\ALSBoat_DemoPython.AdvancedSettings.ini;PythonController."
		
	result = TestContext.client.request_load_situation('DemoPythonController')
		
	remoteControlSocket = ALSLib.TCPClient.TCPClient('127.0.0.1', 7700, 5 )
	remoteControlSocket.connect(5)
	
	for i in range(0,5):
		string = 'setControl t:0 s:0 b:0 h:0 a0:%f a1:%f' %( 1500, 0)
		
		remoteControlSocket.write( string.encode('UTF-8') )
		time.sleep(1)
	
	string = 'setControl t:0 s:0 b:0 h:0 a0:%f a1:%f' %( 0, 20)
	
	remoteControlSocket.write( string.encode('UTF-8') )
	time.sleep(10)

	remoteControlSocket.disconnect()

def RemoteControlScenario():
	TestContext.client.connect()

	DemoVelocity()

	result = TestContext.client.request_toggle_pause()
	result = TestContext.client.request_destroy_situation()
	result = TestContext.client.request_toggle_pause()



print("Environment RemoteControl ")
RemoteControlScenario()



print("--- end of script ---")
