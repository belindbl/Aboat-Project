import time
import threading
import ALSLib.TCPClient, ALSLib.ALSClient 
import ALSLib.ALSHelperFunctionLibrary as ALSFunc


###############################################################################
# This demo shows how to collect the AIS data
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


if __name__ == "__main__":
	TestContext.client.connect()

	result = TestContext.client.request_load_scenario('OpenOceanNaked')
	SettingsPath = "EgoVehicleSettings\\ALSBoat.AdvancedSettings.ini;ForceControlMode."
	
	overrides = ""
	overrides += SettingsPath + "EnableForceControlMode;True\n"
	overrides = ALSFunc.ConvertStringToParameterFormat(overrides)
	TestContext.client.request_load_situation_with_overrides("VelocityControlModeALSBoat", overrides)

	remoteControlSocket = ALSLib.TCPClient.TCPClient('127.0.0.1', 7700, 5 )
	remoteControlSocket.connect(5)
	
	string = 'setControl t:0 s:0 b:0 h:0 a0:%f a1:%f a2:%f a3:%f' %( 0, 0, 30000, 0 )
	
	remoteControlSocket.write( string.encode('UTF-8') )
	time.sleep(5)

	string = 'setControl t:0 s:0 b:0 h:0 a0:%f a1:%f a2:%f a3:%f' %( 0, 0, 0, 3000000 )
	
	remoteControlSocket.write( string.encode('UTF-8') )
	time.sleep(5)

	string = 'setControl t:0 s:0 b:0 h:0 a0:%f a1:%f a2:%f a3:%f' %( 30000, 0, 0, 0 )
	
	remoteControlSocket.write( string.encode('UTF-8') )
	time.sleep(5)

	string = 'setControl t:0 s:0 b:0 h:0 a0:%f a1:%f a2:%f a3:%f' %( 0, 30000, 0, 0 )
	
	remoteControlSocket.write( string.encode('UTF-8') )
	time.sleep(5)

	string = 'setControl t:0 s:0 b:0 h:0 a0:%f a1:%f a2:%f a3:%f' %( 0, 0, 0, 0 )
	
	remoteControlSocket.write( string.encode('UTF-8') )
	time.sleep(5)
	


	string = 'setControl t:0 s:0 b:0 h:0 a0:%f a1:%f a2:%f a3:%f' %( 0, 0, 0, 0 )
	
	remoteControlSocket.write( string.encode('UTF-8') )
	time.sleep(5)
	TestContext.client.execute_destroy_situation()
