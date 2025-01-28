import threading
import ALSLib.TCPClient, ALSLib.ALSClient
import ALSLib.ALSHelperFunctionLibrary as ALSFunc
import ALSLib.ALSHelperImageLibrary as ALSImg

########################
# In this demo we load a situation that has a vehicle and a camera, then 
# we connect to the camera stream and display it in another window using OpenCV
########################

HOST = '127.0.0.1'

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


class TestContext:
	lock = threading.Lock()
	testEnded = False	
	simulation_control_port = 9000
	client = ALSLib.ALSClient.Client((HOST, simulation_control_port),myMessageHandler)


if __name__ == "__main__":
	TestContext.client.connect()
	TestContext.client.request_load_scenario('Default_Scenario')
	overrides = "EgoVehicleSettings\DemoAllSensors.SensorProfile.ini;SensorGroups.GroupData.[0].StreamToNetwork;True"
	TestContext.client.request_load_situation_layer_with_overrides('DemoSensors', overrides)

	Stereo_camera_port = 8890
	print( "Connecting sensor socket to  " + HOST + " " + str(Stereo_camera_port) )
	client = ALSLib.TCPClient.TCPClient(HOST, Stereo_camera_port, 5 )
	client.connect(5)

	imageNum = 0
	while(imageNum < 100):
		data = client.read()
		index = 0
		group_sensor_amount, index = ALSFunc.ReadUint32(data, index)
		print("group amount:", str(group_sensor_amount))

		recieved_images = []
		for i in range(group_sensor_amount):
			sensor_type, index 		= ALSFunc.ReadString(data, index)
			sensor_path, index 		= ALSFunc.ReadString(data, index)
			print("sensor ", i,": Type ", sensor_type, "  path:", sensor_path)

			image, index, image_width, image_height	= ALSFunc.ReadImage_Group(data, index)
			recieved_images.append(image)

			extra_string, index 	= ALSFunc.ReadString(data, index)
			print("extra sting: ", extra_string)
			
		stack = ALSImg.StackImages(recieved_images)
		ALSImg.JustDisplay(stack)
