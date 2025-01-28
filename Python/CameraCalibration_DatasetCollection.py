
from io import BytesIO
import time, threading, struct, json
from os import mkdir as osmkdir, path as ospath
from shutil import rmtree
from PIL import Image
import numpy as np
#import array as nparray, save as npsave, dtype as npdtyp, frombuffer as npfrombuffer, reshape as npreshape
from ALSLib.ALSTestManager import ALSTestManager
from ALSLib.ALSTestManager import safe_print 
import ALSLib.ALSClient 
import ALSLib.TCPClient
import ALSLib.ALSHelperFunctionLibrary as ALSFunc
import ALSLib.ALSHelperImageLibrary as ALSImg

###############################################################################
class VehicleStatus:
	def __init__(self):
		self.SimulationTime = 0.0
		self.StatusString = ''
		self.CurrentDestination = [0.0,0.0,0.0]
		self.CurrentPosition = [0.0,0.0,0.0]
		self.HasDestination = False
	def Copy(self, other):
		self.SimulationTime = other.SimulationTime
		self.StatusString = other.StatusString
		self.CurrentDestination = other.CurrentDestination
		self.CurrentPosition = other.CurrentPosition
		self.HasDestination = other.HasDestination

###############################################################################
def myMessageHandler(rawMessage):
	strMessage = rawMessage.decode('utf-8')

	cmdList = strMessage.split(" ")
	if cmdList[0].startswith("EndCondition") :
		print ("Trigger received: ", cmdList[1])
		if( cmdList[1].startswith("TimeEnded")):
			TestManager.lock.acquire()
			TestManager.testEnded = True
			print("setting TestEnded")
			TestManager.lock.release()

	if strMessage.startswith("Status"):
		cmdList = strMessage.split(" ",2)
		TestManager.lock.acquire()
		TestManager.vehicleStatus.SimulationTime = float(cmdList[1])
		TestManager.vehicleStatus.StatusString = cmdList[2]
		TestManager.lock.release()


###############################################################################
class SensorDataThread( threading.Thread):
	def __init__(self, ThreadID, client, function, prefix):
		threading.Thread.__init__(self)
		self.threadID = ThreadID
		self.client = client
		self.function = function
		self.prefix = prefix
	def run(self):
		print ("Starting ", self.threadID,"\n")
		self.function(self.client, self.threadID, self.prefix)
		print ("Exiting ", self.threadID,"\n")

###############################################################################
class ThreadedSensorRecieve:
	def __init__(self, sensorDefinitions):
		self.sensorThreads = []
		[self.addSensor(sensor[0], sensor[1], sensor[2], sensor[3])
		 	for sensor in sensorDefinitions]
		self.startAllThreads()
	def addSensor(self, socketAddress, socketPort, recieveFunction, prefix):
		socket = ALSLib.TCPClient.TCPClient(socketAddress, socketPort, 15 )
		socket.connect(10)
		thread = SensorDataThread(len(self.sensorThreads), socket, recieveFunction, prefix)
		self.sensorThreads.append([socket, thread])
	def startAllThreads(self):
		[thread[1].start() for thread in self.sensorThreads]
	def waitAllThreads(self):
		[thread[1].join() for thread in self.sensorThreads ]


class ThreadedWriteFile( threading.Thread):
	def __init__(self, array, timestamp, prefix, ID,imageNum, lossless:bool=False):
		threading.Thread.__init__(self)
		self.array = array
		self.timestamp = timestamp
		self.prefix = prefix
		self.ID = ID
		self.imageNum = imageNum
		self.lossless = lossless
	def run(self):
		WriteImage(self.array, self.timestamp, self.prefix,self.ID, self.imageNum, self.lossless)



def ReadUint32(buffer, index):
	return (int(struct.unpack('<L', buffer[index:index+4])[0]), index + 4)


def DecodeImageData(image_data, image_width, image_height, image_channels):
	array = np.frombuffer(image_data, dtype=np.dtype("uint8"))
	array = np.reshape(array, (image_height, image_width, image_channels))
	return array



###############################################################################
def receiveImage(client, ID, prefix):
	imageNum = -1
	time.sleep(0.5)
	while True:
		try:
			data = client.read()
		except Exception as e:
			safe_print('an exception occured while reading the next sensor data')
			safe_print(e)
			TestManager.testEnded = True
			TestManager.result = ALSTestManager.ALS_ERROR
			break

		index = 0
		imageNum += 1
		img, index, width, height = ALSFunc.ReadImage_Stream(data, index)

		parsed_json = ''
		if index < len(data) :
			extra_string, index = ALSFunc.ReadString(data,index)
			parsed_json 		= json.loads(extra_string)
		else:
			safe_print("No Meta data in this stream")
		safe_print(parsed_json)
		safe_print ("image ", imageNum," received! - thread ",ID, " timestamp ",round(float(parsed_json['T']),3))

		TestManager.lock.acquire()
		if ID == 0:
			TestManager.LastImageTimeStamp = round(float(parsed_json['T']),3)
			TestManager.LastImage = img
			TestManager.LastImageMetaData = parsed_json
		if ID == 1:
			TestManager.LastGTTimeStamp = round(float(parsed_json['T']),3)
			TestManager.LastGT = img
			TestManager.LastGTMetaData = parsed_json 
		TestManager.lock.release()

		time.sleep(0.01)

		if TestManager.testEnded :
			break
		if imageNum>SAFETYNET_KILL_NUMIMAGES :
			TestManager.testEnded = True
			break


###############################################################################
def receiveDepth(client, ID, prefix):
	imageNum = -1
	time.sleep(1)
	while True:
		try:
			data = client.read()
		except Exception as e:
			safe_print('an exception occured while reading the next sensor data')
			safe_print(e)
			TestManager.testEnded = True
			TestManager.result = ALSTestManager.ALS_ERROR
			break
		imageNum += 1

		index = 0
		getval = lambda index: struct.unpack('<L', data[index*4:index*4+4])[0]
		image_height = getval(index)
		image_width = getval(index+1)
		image_channels = getval(index+2)
		image_data = data[3*4:4*3+image_height*image_width*image_channels]
		array = np.frombuffer(image_data, dtype=np.dtype("uint8"))
		array = np.reshape(array, (image_height, image_width, image_channels))

		currentOffset = 4*3+image_height*image_width*image_channels        
		currentOffset += 4
		codedInfo = data[currentOffset:-2]
		databox = codedInfo.decode('UTF-8')[:]
		parsed_json = json.loads(databox)

		safe_print ("image ", imageNum," received! - thread ",ID, " timestamp ",round(float(parsed_json['T']),3))

		TestManager.lock.acquire()
		TestManager.LastDepthTimeStamp = round(float(parsed_json['T']),3)
		TestManager.LastDepth = array
		TestManager.lock.release()
		
		time.sleep(0.01)
		
		if imageNum > SAFETYNET_KILL_NUMIMAGES :
			TestManager.testEnded = True
			break

		if TestManager.testEnded :
			break
import cv2
def WriteImage(array, timestamp, prefix, ID,imageNum, lossless:bool=False):
	im = Image.fromarray(array)
	b, g, r, a = im.split()
	im = Image.merge("RGB", (r, g, b))
	timeStampStr = '%.2f'%(timestamp)
	filename = ALSFunc.get_sensordata_path(TestManager.test_id+'/'+ str(ID) + '_' + prefix  + '_' + str(imageNum) + '_' + timeStampStr)
	im.save(filename+'.png', 'png' , optimize=False, compress_level=0)


def WriteMetaData(text, timestamp, prefix, ID, imageNum):
	timeStampStr = '%.2f'%(timestamp)
	filename = ALSFunc.get_sensordata_path(TestManager.test_id+'/'+ str(ID) + '_' + prefix  + '_' + str(imageNum) + '_' + timeStampStr)
	with open(filename+'.txt',"w") as text_file:
		text_file.write(json.dumps(text))
	

def WriteDepth(array, timestamp, prefix, ID,imageNum):
	array_b, array_g, array_r = array[:, :, 0], array[:, :, 1], array[:, :, 2]
	arraybf = np.array(array_b,'float32')
	arrayrf = np.array(array_r,'float32')
	arraygf = np.array(array_g,'float32')
	farPlaneDist = 50.0
	array24 = (((arraybf*65535.0 + arraygf*256.0+ arrayrf)/16777216.0)**2)*farPlaneDist

	timeStampStr = '%.2f'%(timestamp)
	filename = ALSFunc.get_sensordata_path(TestManager.test_id+'/'+ str(ID) + '_' + prefix  + '_' + str(imageNum) + '_' + timeStampStr)
	np.save(filename+'.npy', array24)
	greyimage = Image.fromarray(array_b)
	greyimage.save(filename+'_greyb.png','png')


SAFETYNET_KILL_NUMIMAGES=2500
NUM_IMAGE_COLLECTED=20
###############################################################################
def CollectDataset():
	TestFolderPath = ALSFunc.get_sensordata_path(TestManager.test_id)
	if( ospath.exists(TestFolderPath)):
		rmtree(TestFolderPath, ignore_errors=True)
		time.sleep(1)
	try :
		osmkdir(ALSFunc.get_sensordata_path())
	except Exception as e:
		safe_print(e)
	try :
		osmkdir(TestFolderPath)
	except  Exception as e:
		safe_print(e)

		

	time.sleep(0) # give some time for the pile to stabilize
	safe_print("Starting the sensors collection Threads")
	sensorDefinintion = [
		[TestManager.details['InstanceIP'], 8880, receiveImage, "Disto"],
		[TestManager.details['InstanceIP'], 8881, receiveImage, "NoDisto"]
	]
	sensorThread = ThreadedSensorRecieve(sensorDefinintion)
	time.sleep(1) # give some time for the pile to stabilize

	TestManager.result = ALSTestManager.ALS_SUCCESS
	count = 0
	TestManager.timeStampAfter = 3.5
	LastTimeStamp = 0
	collection_started = False
	StartedProcesses = []
	while True:
		
		if not collection_started and TestManager.LastImageTimeStamp > TestManager.timeStampAfter:
			collection_started = True
		
		if	collection_started and\
			TestManager.LastImageTimeStamp != LastTimeStamp and \
			TestManager.LastGTTimeStamp == TestManager.LastImageTimeStamp :
			LastTimeStamp = TestManager.LastImageTimeStamp
			
			#async writing makes it faster and allow this script to keep collecting data
			TestManager.lock.acquire()
			img1 = TestManager.LastImage
			img2 = TestManager.LastGT
			TestManager.lock.release()
			
			p = ThreadedWriteFile(img1, TestManager.LastImageTimeStamp, "Disto", 0, count)
			p.start()
			StartedProcesses.append(p)
			p = ThreadedWriteFile(img2, TestManager.LastImageTimeStamp, "NoDisto", 1, count)
			p.start()
			StartedProcesses.append(p)
			WriteMetaData(TestManager.LastImageMetaData,TestManager.LastImageTimeStamp, "Disto", 0, count)
			WriteMetaData(TestManager.LastGTMetaData,TestManager.LastImageTimeStamp, "NoDisto", 1, count)

			print("saving ID:"+ str(count) +"with Timestamp "+ str(LastTimeStamp))
			count+=1
			if count > NUM_IMAGE_COLLECTED:
				TestManager.testEnded = True
		
		TIMEOUT_OUT_OF_SYNC = 15
		if TestManager.LastImageTimeStamp - TIMEOUT_OUT_OF_SYNC > LastTimeStamp :
			safe_print("Timeout, threads out of sync for too long")
			TestManager.result = ALSTestManager.ALS_ERROR
			TestManager.lock.acquire()
			TestManager.testEnded = True
			TestManager.lock.release()

		if 	TestManager.testEnded == True:
			safe_print( 'end of test recieved')
			break

		#print(":")
		time.sleep(0.01)
		#print(">")


	sensorThread.waitAllThreads()
	for P in StartedProcesses:
		P.join()

	#sometimes it might disconnect so just reconnect in case..
	TestManager.client.connect()

	# Remember to keep this "SaveReplayFiles()" if you want to see what happens for tests that failed. (you can comment it for local tests)
	#also this needs to be called before the scenario is stopped.. 
	TestManager.StopScenario()
	TestManager.SaveReplayFiles()

	#upload all files to s3
	print("Saving:")
	TestManager.SaveSensorDataFilesToS3(TestFolderPath)
	
	if(TestManager.test_id != 'test_id_local' and ospath.exists(TestFolderPath)):
		rmtree(TestFolderPath, ignore_errors=True)

	sensorThread.waitAllThreads()

	# This result can be set in the message handler function at the top of the file
	return TestManager.result


###############################################################################
# The TestManager is the central part of the testing system.
# We give the message handler function to receive the data
TestManager = ALSTestManager(myMessageHandler)
TestManager.vehicleStatus = VehicleStatus()
TestManager.LastImageTimeStamp = 0
TestManager.LastGTTimeStamp = 0
TestManager.LastDepthTimeStamp = 0

TestManager.StartTest(CollectDataset)

if __name__ == "__main__":
	print("wrong")
print("--- end of script ---")
