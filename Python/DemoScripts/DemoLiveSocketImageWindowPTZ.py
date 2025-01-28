import threading
from math import sin, cos
from time import time
import ALSLib.TCPClient, ALSLib.ALSClient
import ALSLib.ALSHelperFunctionLibrary as ALSFunc
import ALSLib.ALSHelperImageLibrary as ALSImg
########################
# In this demo we load a situation that has a vehicle and a camera, then
# we connect to the camera stream and display it in another window using OpenCV
########################

def lerp(a: float, b: float, t: float) -> float:
    return (1 - t) * a + t * b

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


HOST = '127.0.0.1'

class TestContext:
	lock = threading.Lock()
	testEnded = False
	simulation_control_port = 9000
	client = ALSLib.ALSClient.Client((HOST, simulation_control_port),myMessageHandler)


if __name__ == "__main__":
	TestContext.client.connect()
	TestContext.client.request_load_scenario('Default_Scenario')
	overrides = "EgoVehicleSettings\DemoAllSensors.SensorProfile.ini;Sensors.Sensors.[0].StreamToNetwork;True"
	TestContext.client.request_load_situation_layer_with_overrides('DemoSensors', overrides)

	#The camera port is defined inside the situation ini files, can be changed in Sensor Editor.
	CAMERA_PORT = 8880
	print( "Connecting sensor socket to  " + HOST + " " + str(CAMERA_PORT) )
	client = ALSLib.TCPClient.TCPClient(HOST, CAMERA_PORT, 5 )
	client.connect(5)


	imageNum = 0
	while(True):
		data = client.read()
		phase = (sin(time()*0.5)+1.0)*0.5
		phase2 = (cos(time())+1.0)*0.5
		pan = lerp(-90.0, 90.0, phase)
		tilt = lerp(-20.0, 20.0, phase2)
		zoom = lerp(-1.0, 1.0, phase2)

		# pan = 0
		# tilt = 0
		# zoom = 0

		TestContext.client.request(f"SetPTZOffset ego Sensors.[0] {pan} {tilt} {zoom}")

		index = 0
		imageNum += 1
		img, index, width, height = ALSFunc.ReadImage_Stream(data, index)

		ALSImg.JustDisplay(img)
