import threading
import ALSLib.TCPClient, ALSLib.ALSClient


###############################################################################
# This demo shows how to collect the ARPA data
###############################################################################

#HOST = '127.0.0.1'
HOST = 'localhost'

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
	TestContext.client.request_load_situation('ARPA')
	print("Loaded Situation...")
	arpa_port = 8880
	client = ALSLib.TCPClient.TCPClient(HOST, arpa_port,15)
	client.connect(5)

	while(True):
		data = client.read()
		index = 0
		data_str = data.decode('utf8')
		print(data_str)
