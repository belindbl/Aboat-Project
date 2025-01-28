import threading
import ALSLib.ALSClient
import ALSLib.TCPClient
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
	TestContext.client.request_load_situation('AISExample')

	ais_port = 8880
	client = ALSLib.TCPClient.TCPClient(HOST, ais_port, 5)
	client.connect(5)

	while(True):
		data = client.read()
		
		index = 0
		group_sensor_amount, index = ALSFunc.ReadUint32(data, index)

		PCpoints = []
		received_images = []

		for i in range(group_sensor_amount):
			sensor_type, index 		= ALSFunc.ReadString(data, index)
			sensor_path, index 		= ALSFunc.ReadString(data, index)
			
			ais_reading, index = ALSFunc.ReadString(data, index)

			if len(ais_reading) > 0:
				print(ais_reading)
