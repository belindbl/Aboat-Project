import time
import threading
import logging
import ALSLib.TCPClient
import ALSLib.ALSHelperFunctionLibrary as ALSFunc
import ALSLib.ALSClient 

###############################################################################
# Here we show how you can hook into the remote control port of the vehicles 
# and send commands to it. 
# Each vehicles has its own control socket that is defined in the details of 
# situation file. 
# Basic vehicles have Throttle, Steerin, Brake type of control, but
# optional list of axis are also available for more complex vehicles. 
# An optional dt command can be used to pause the simulationa automatically
#
# This also shows the status message that is recieved by the vehilce
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
	client = ALSLib.ALSClient.Client((localhost, echo_port), myMessageHandler)
	PORT = 7700
	HOST = '127.0.0.1'
	vehicleStatus = VehicleStatus()

def RemoteControlScenario():
	TestContext.client.connect()

	result = TestContext.client.request_load_scenario('OpenOceanNaked')

	SettingsPath = "EgoVehicleSettings\\ALSBoat.AdvancedSettings.ini;VelocityControlModePID."
	
	overrides = ""
	overrides += SettingsPath + "EnableVelocityControlMode;True\n"
	overrides += SettingsPath + "UseAngularVelocity;False\n"

	overrides = ALSFunc.ConvertStringToParameterFormat(overrides)

	result = TestContext.client.request_load_situation_with_overrides("VelocityControlModeALSBoat", overrides)
		
	remoteControlSocket = ALSLib.TCPClient.TCPClient('127.0.0.1', 7700, 5 )
	remoteControlSocket.connect(5)
	
	throttle = 0.0
	steering = 0.0
	brake = 0.0
	handbrake = 0.0

	TargetSpeed = 30.0
	TargetHeading = 45.0
	TargetStrafe = 0.0

	string = 'setControl t:%f s:%f b:%f h:%i a0:%f a1:%f a2:%f' %( throttle, steering, brake, handbrake, TargetSpeed, TargetHeading, TargetStrafe)
	
	remoteControlSocket.write( string.encode('UTF-8') )
	time.sleep(10)

	string = 'setControl t:%f s:%f b:%f h:%i a0:%f a1:%f a2:%f' %( throttle, steering, brake, handbrake, TargetSpeed, -TargetHeading, -TargetStrafe)
	
	remoteControlSocket.write( string.encode('UTF-8') )
	time.sleep(10)

	string = 'setControl t:%f s:%f b:%f h:%i a0:%f a1:%f a2:%f' %( throttle, steering, brake, handbrake, TargetSpeed, 135, TargetStrafe)
	
	remoteControlSocket.write( string.encode('UTF-8') )
	time.sleep(10)

	string = 'setControl t:%f s:%f b:%f h:%i a0:%f a1:%f a2:%f' %( throttle, steering, brake, handbrake, TargetSpeed, -135, -TargetStrafe)
	
	remoteControlSocket.write( string.encode('UTF-8') )
	time.sleep(10)

	result = TestContext.client.request_toggle_pause()
	result = TestContext.client.request_destroy_situation()
	result = TestContext.client.request_toggle_pause()


print("Environment RemoteControl ")
RemoteControlScenario()

print("--- end of script ---")
