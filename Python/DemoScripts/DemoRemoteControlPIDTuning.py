import json
import time
import threading
import logging
import ALSLib.ALSClient 
import ALSLib.TCPClient
import ALSLib.ALSHelperFunctionLibrary as ALSFunc

###############################################################################
# Here are some handy overrides for PID tuning. Tuning mode is enabled by the
# override, which adds the PID values to the status message JSON. PID coeffs
# can also be changed with the override, which then allows doing the whole 
# tweaking process in this file, without having to jump between this file and
# the vehicle advanced parameters file.
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
	string = rawMessage.decode('utf-8')
	
	cmdList = string.split(" ")
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
		status = json.loads(TestContext.vehicleStatus.StatusString)
		PID = status.get("PID")

		TuningMessage = f""
		TuningMessage += f"PThrottle {PID.get('PThrottle')}\n"
		TuningMessage += f"IThrottle {PID.get('IThrottle')}\n"
		TuningMessage += f"DThrottle {PID.get('DThrottle')}\n"
		TuningMessage += f"PAngular {PID.get('PAngular')}\n"
		TuningMessage += f"IAngular {PID.get('IAngular')}\n"
		TuningMessage += f"DAngular {PID.get('DAngular')}\n"
		TuningMessage += f"PStrafe {PID.get('PStrafe')}\n"
		TuningMessage += f"IStrafe {PID.get('IStrafe')}\n"
		TuningMessage += f"DStrafe {PID.get('DStrafe')}\n"
	
		print(TuningMessage)
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
	
	SettingsPath = "EgoVehicleSettings\\ALSBoat.AdvancedSettings.ini;VelocityControlModePID."
	
	overrides = ""
	overrides += SettingsPath + "EnableTuningMode;True\n"
	overrides += SettingsPath + "EnableVelocityControlMode;True\n"
	overrides += SettingsPath + "UseAngularVelocity;True\n"

	overrides += SettingsPath + "ThrottleProportionalCoeff;0.01\n"
	overrides += SettingsPath + "ThrottleIntegralCoeff;0.05\n"
	overrides += SettingsPath + "ThrottleDerivativeCoeff;0.0\n"
	overrides += SettingsPath + "ThrottleBias;0.0\n"

	overrides += SettingsPath + "AngularProportionalCoeff;0.1\n"
	overrides += SettingsPath + "AngularIntegralCoeff;0.06\n"
	overrides += SettingsPath + "AngularDerivativeCoeff;0.001\n"
	overrides += SettingsPath + "AngularBias;0.0\n"

	overrides += SettingsPath + "StrafeProportionalCoeff;0.2\n"
	overrides += SettingsPath + "StrafeIntegralCoeff;0.00001\n"
	overrides += SettingsPath + "StrafeDerivativeCoeff;0.00001\n"
	overrides += SettingsPath + "StrafeBias;0.0\n"

	overrides = ALSFunc.ConvertStringToParameterFormat(overrides)

	result = TestContext.client.request_load_situation_with_overrides("VelocityControlModeALSBoat", overrides)

	remoteControlSocket = ALSLib.TCPClient.TCPClient('127.0.0.1', 7700, 5 )
	remoteControlSocket.connect(5)

	TargetSpeed = 5.0
	TargetAngularVelocity = 0.0
	TargetStrafe = 0

	string = 'setControl t:%f s:%f b:%f h:%i a0:%f a1:%f a2:%f' %( 0.0, 0.0, 0.0, 0.0, TargetSpeed, TargetAngularVelocity, TargetStrafe)
	
	remoteControlSocket.write( string.encode('UTF-8') )

	time.sleep(1000)

	result = TestContext.client.request_toggle_pause()
	result = TestContext.client.request_destroy_situation()
	result = TestContext.client.request_toggle_pause()


print("Environment RemoteControl ")
RemoteControlScenario()

print("--- end of script ---")
