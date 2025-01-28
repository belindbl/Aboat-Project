import threading, time, json
import ALSLib.TCPClient, ALSLib.ALSClient
import ALSLib.ALSHelperFunctionLibrary as ALSFunc
import ALSLib.ALSHelperImageLibrary as ALSImg

########################
# In this demo we load a situation that has a vehicle and a camera, then 
# we connect to the camera stream and display it in another window using OpenCV
########################



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

	# Get the sensor list
	time.sleep(1)
	sensorlist = TestContext.client.get_sensor_list()
	parsed_json = json.loads(sensorlist)

	# Find the host and port of the camera
	for x in parsed_json['sensors']:
		if x['path'] == 'Sensors.[0]':
			sensor = x
			break
	camera_port = sensor['sensor_port']
	host = sensor['sensor_ip']

	#The camera port is defined inside the situation ini files, can be changed in Sensor Editor. 
	print( "Connecting sensor socket to  " + host + " " + str(camera_port) )
	client = ALSLib.TCPClient.TCPClient(host, camera_port, 5 ) # Arg #3 is timeout in seconds
	client.connect(5)

	imageNum = 0
	while(imageNum < 100):
		data = client.read()

		index = 0
		imageNum += 1
		img, index, width, height = ALSFunc.ReadImage_Stream(data, index)
		
		ALSImg.JustDisplay(img)
