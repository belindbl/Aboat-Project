import traceback, time, zipfile, datetime, threading, json, inspect, math, struct, signal, copy, cv2, pathlib, random
from typing import Type
from enum import Enum
from os import mkdir as osmkdir, makedirs as osmakedirs, path as ospath, remove as removefile, listdir, sep as ospath_sep
from shutil import rmtree
from PIL import Image
from typing import List
import numpy as np
from .ALSTestManager import ALSTestManager, safe_print
import ALSLib.ALSHelperImageLibrary as ALSImg
import ALSLib.ALSHelperFunctionLibrary as ALSFunc
from . import TCPClient2, ALSClient
from abc import ABC, abstractmethod
from multiprocessing import Process, Queue, shared_memory, Pool
from queue import Empty as QueueEmpty
from .ALSReceiver import alsreceiver

###############################################################################
## Thread defintions
###############################################################################

class ThreadSafeData:
	def __init__(self, initializeWith=None, test_manager=None):
		self.data = initializeWith
		self.test_manager = test_manager
	def WriteData(self, data):
		self.test_manager.lock.acquire()
		self.data = data
		self.test_manager.lock.release()


###############################################################################
class SensorDataThread( threading.Thread):
	def __init__(self, ThreadID, client, function, prefix, data:List[ThreadSafeData],data_defintion, test_manager:ALSTestManager):
		threading.Thread.__init__(self)
		self.threadID = ThreadID
		self.client = client
		self.function = function
		self.prefix = prefix
		self.data = data
		self.test_manager = test_manager
		self.data_definition = data_defintion
	def run(self):
		print ("Starting ", self.threadID,"\n")
		self.function(self.client, self.threadID, self.prefix, self.data, self.data_definition, self.test_manager)
		print ("Exiting ", self.threadID,"\n")

###############################################################################
class ThreadedSensorRecieve:
	def __init__(self, sensorDefinitions, test_manager):
		self.sensorThreads = []
		# sensorDefinintion = [
		# 	[(0)TestManager.details['InstanceIP'], (1) 8880, (2)recieveStereoCamera, (3)"StereoPair", 
		# 				(4)[LastTime, LastImageArray, LastMetaDataArray], (5) dataDefinition]
		[self.addSensor(sensor[0], sensor[1], sensor[2], sensor[3], sensor[4], sensor[5],test_manager)
		 	for sensor in sensorDefinitions]
		self.startAllThreads()
	def addSensor(self, socketAddress, socketPort, recieveFunction, prefix, data, data_definition, test_manager):
		socket = TCPClient2.TCPClient(socketAddress, socketPort, 15 )
		socket.connect(10)
		thread = SensorDataThread(len(self.sensorThreads), socket, recieveFunction, prefix, data, data_definition, test_manager)
		self.sensorThreads.append([socket, thread])
	def startAllThreads(self):
		[thread[1].start() for thread in self.sensorThreads]
	def waitAllThreads(self):
		[thread[1].join() for thread in self.sensorThreads ]

###############################################################################
class ThreadedWriteDataToFile( threading.Thread):
	def __init__(self, array, timestamp, prefix, ID,imageNum, func, base_path:str="", lossless:bool=False):
		threading.Thread.__init__(self)
		self.array = array
		self.timestamp = timestamp
		self.prefix = prefix
		self.ID = ID
		self.imageNum = imageNum
		self.lossless = lossless
		self.func = func
		self.base_path = base_path
	def run(self):
		self.func(self.array, self.timestamp, self.prefix,self.ID, self.imageNum, self.base_path, self.lossless)


###############################################################################
#this is a rolling thread structure (thread are closed and new are added, every now and then )
class ThreadedWriting:
	def __init__(self, dataDefinitions, count, base_path:str=""):
		self.WritingThreads = []
		self.count = count
		for curr_data in dataDefinitions:
			for id, dataDefinition in enumerate(curr_data[1]):
				# print(f"{curr_data}, {id}")
				self.addWritingData(curr_data[0].data[id], dataDefinition[3], dataDefinition[0].data,
								 dataDefinition[1], dataDefinition[2], base_path)

		self.startAllThreads()
	def addWritingData(self, dataToWrite, writing_function, timestamp, prefix, ID, base_path:str="", lossless:bool = True):# data, timestamp, prefix, ID, imageNum, lossless=True):
		thread = ThreadedWriteDataToFile(dataToWrite, timestamp,prefix, ID, self.count, writing_function, base_path=base_path, lossless=lossless)
		self.WritingThreads.append(thread)
	def startAllThreads(self):
		[thread.start() for thread in self.WritingThreads]
	def waitAllThreads(self):
		[thread.join() for thread in self.WritingThreads ]

class ThreadedWritingRadar:
	def __init__(self, dataDefinitions, count, base_path:str=""):
		self.WritingThreads = []
		self.count = count
		for curr_data in dataDefinitions:
			for id, dataDefinition in enumerate(curr_data[1]):
				self.addWritingData(curr_data[0].data, dataDefinition[3], dataDefinition[0].data,
								 dataDefinition[1], dataDefinition[2], base_path)

		self.startAllThreads()
	def addWritingData(self, dataToWrite, writing_function, timestamp, prefix, ID, base_path:str="", lossless:bool = True):# data, timestamp, prefix, ID, imageNum, lossless=True):
		thread = ThreadedWriteDataToFile(dataToWrite, timestamp,prefix, ID, self.count, writing_function, base_path=base_path, lossless=lossless)
		self.WritingThreads.append(thread)
	def startAllThreads(self):
		[thread.start() for thread in self.WritingThreads]
	def waitAllThreads(self):
		[thread.join() for thread in self.WritingThreads ]

###############################################################################
## WRITING FUNCTIONS
###############################################################################

def WriteMetaData(texts:str, timestamp, prefix, ID, imageNum, base_path:str="", lossless=True):
	filename = f'{base_path}/Generation/{str(ID)}_{prefix}_{str(imageNum)}'
	#base_path+'/Generation/'+ str(ID) + '_' + prefix + '_' + str(imageNum)
	with open(filename+'.txt',"w") as text_file:
		text_file.write(json.dumps(texts))


def WriteDepth(array, timestamp, prefix, ID,imageNum, base_path:str="", lossless:bool = True, write_grey_image:bool = False):
	array_b, array_g, array_r = array[:, :, 0], array[:, :, 1], array[:, :, 2]
	arraybf = np.array(array_b,'float32')
	arrayrf = np.array(array_r,'float32')
	arraygf = np.array(array_g,'float32')
	farPlaneDist = 100.0
	array24 = (((arraybf*float(0xFFFF) + arraygf*float(0xFF)+ arrayrf)/float(0xFFFFFF))**2)*farPlaneDist

	#store the zip file (first npy then zip, then delete npy)
	filename = f'{base_path}/Generation/{str(ID)}_{prefix}_{str(imageNum)}'
	#base_path+'/Generation/'+ str(ID) + '_' + prefix  + '_' + str(imageNum)
	np.save(filename+'.npy', array24)
	zipfilename = f'{base_path}/Generation/{str(ID)}_{prefix}_grey'
	#base_path+'/Generation/'+ str(ID) + '_' + prefix  + '_grey'
	with zipfile.ZipFile(zipfilename+'.zip','a') as zipf:
		#zipf.writestr(ospath.basename(filename+'.npy'), array24)
		zipf.write(filename+'.npy',ospath.basename(filename+'.npy'), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
	import os
	os.remove(filename+'.npy')

	#store grey for video
	if write_grey_image == True:
		greyimage = Image.fromarray(array_b)
		# filename = base_path+'/Generation/'+ str(ID) + '_' + prefix  + '_grey_' + str(imageNum)
		filename = f'{base_path}/Generation/{str(ID)}_{prefix}_grey_{str(imageNum)}'
		greyimage.save(filename+'.png','png')


def WriteDepthWithGrey(array, timestamp, prefix, ID,imageNum, base_path:str="", lossless:bool = True):
	WriteDepth(array, timestamp, prefix, ID,imageNum, lossless, base_path=base_path, lossless=lossless)


def WriteImage(array, timestamp, prefix, ID,imageNum, base_path:str="", lossless:bool=False):
	im = Image.fromarray(array)
	b, g, r, a = im.split()
	im = Image.merge("RGB", (r, g, b))
	timeStampStr = '%.2f'%(timestamp)
	# filename = BASE_DIRECTORY+TestManager.test_id+'/'+ str(ID) + '_' + prefix  + '_' + str(imageNum) + '_' + timeStampStr
	# filename = base_path+'/Generation/'+ str(ID) + '_' + prefix  + '_' + str(imageNum)
	filename = f'{base_path}/Generation/{str(ID)}_{prefix}_{str(imageNum)}'
	im.save(filename+'.png', 'png' , optimize=False, compress_level=9)


def WriteImageToVideo(array, timestamp, prefix, ID,imageNum, base_path:str="", lossless:bool=False, write_file:bool=False):
	global video_streaming_processes, FPS
	entry_key = str(ID) + '_' + prefix
	if not entry_key in video_streaming_processes.keys():
		filename = f'{base_path}/Videos/{str(ID)}_{prefix}.avi'
		# p = Popen(['ffmpeg', '-y', '-f', 'image2pipe',  '-r', str(FPS),
		# 	'-i', '-', '-vcodec', 'mpeg4', '-q:v', '5', '-b:v', '100M', '-r',
		# 	str(FPS) , filename], stdin=PIPE)
		p = ALSImg.ImageWriterVideoStream(filename, FPS)
		video_streaming_processes[entry_key] = p
	proc = video_streaming_processes[entry_key]
	#proc.add_image_from_array(array)

	im = Image.fromarray(array)
	b, g, r, a = im.split()
	im = Image.merge("RGB", (r, g, b))
	#im.save(proc.stdin, 'png')
	try:
		proc.add_image(im)
	except Exception:
		print( 'couldnt write to video')

	if (write_file):
		filename = f'{base_path}/Generation/{str(ID)}_{prefix}_{str(imageNum)}'
		im.save(filename+'.png', 'png' , optimize=False, compress_level=9)

def WriteImageBoth(array, timestamp, prefix, ID,imageNum, base_path:str="", lossless:bool=False):
	WriteImageToVideo(array, timestamp, prefix, ID,imageNum, lossless, base_path=base_path, write_file = True)

def WriteRadarData(array, timestamp, prefix, ID,frameNum, base_path:str="", lossless:bool=False):
	filename = f'{base_path}/Generation/{str(ID)}_{prefix}_{str(frameNum)}_{(timestamp)}.txt'
	with open(filename, 'w') as file:
		for point in array:
			file.write(point.serialize())
			
def WriteRadarDataDummy(array, timestamp, prefix, ID,frameNum, base_path:str="", lossless:bool=False):
	WriteRadarData(array, timestamp, prefix, ID,frameNum, base_path, lossless)

###############################################################################
###     RecieveFunctions
###############################################################################

def recieveImage(client, ID, prefix, dataDest:List[ThreadSafeData], data_definition, test_manager:ALSTestManager):
	imageNum = -1
	StartedProcesses:List[ThreadedWriting] = []
	time.sleep(0)
	while True:
		data 	= client.read()
		index 	= 0
		img, index, width, height = ALSFunc.ReadImage_Stream(data, index)
		images_to_display = []
		if index < len(data) :
			extra_string, index = ALSFunc.ReadString(data,index)
			parsed_string 		= json.loads(extra_string)
			if "T" in parsed_string.keys():
				time 			= float(parsed_string['T'])
			#tracked_objects =  parsed_string['TrackedObj']
			# for a_obj in tracked_objects:
			# 	print(a_obj)
			# 	if None is not a_obj.get('BB2D'):
			# 		BoundingBox			= a_obj['BB2D']
			# 		pt1 = (int(BoundingBox[0]['X']),int(BoundingBox[0]['Y']))
			# 		pt2 = (int(BoundingBox[1]['X']),int(BoundingBox[1]['Y']))
			# 		img = cv2.rectangle(img, pt1, pt2, (0, 0, 255),2)
			images_to_display.append(img)


		dataDest[0].WriteData(time)
		dataDest[1].WriteData(images_to_display)
		dataDest[2].WriteData(parsed_string)

		if time > test_manager.save_after_time : 
			started = ThreadedWriting(data_definition,imageNum, test_manager.base_path)
			StartedProcesses.append(started)

		#periodically clean the list
		if len(StartedProcesses)>15:
			for i in range(5):
				proc = StartedProcesses.pop(0)
				proc.waitAllThreads()

		time.sleep(0.01)

		if time > test_manager.session_duration:
			safe_print( 'Session Ended')
			test_manager.testEnded = True
			break

		if imageNum > test_manager.safety_net_kill :
			safe_print( 'SAFETY NET KILL')
			test_manager.testEnded = True
			break

		if test_manager.testEnded :
			safe_print( f'Exiting thread {str(ID)}')
			break
	
	#ended -  wait for files written
	for P in StartedProcesses:
		P.waitAllThreads()


"""
def recievePointCloud(client, ID, prefix, dataDest:List[ThreadSafeData], test_manager:ALSTestManager):
	imageNum = -1
	while True:
		data = client.read()
		imageNum += 1

		sizeofFloat = 4
		index = 0
		getFloatVal = lambda index: struct.unpack('<f', data[index*sizeofFloat:index*sizeofFloat+sizeofFloat])[0]

		posX = getFloatVal(index)
		index += 1
		posY=getFloatVal(index)
		index += 1
		posZ= getFloatVal(index)
		index += 1
		quatW= getFloatVal(index)
		index += 1
		quatX= getFloatVal(index)
		index += 1
		quatY= getFloatVal(index)
		index += 1
		quatZ= getFloatVal(index)
		index += 1
		numPoints= getFloatVal(index)
		index += 1
		timeStart= getFloatVal(index)
		index += 1
		timeEnd =  getFloatVal(index)
		index += 1
		numberOfBeams =  getFloatVal(index)
		index += 1


		pointCloudData = data[index*sizeofFloat:]
		array = np.frombuffer(pointCloudData, dtype=np.dtype("float32"))
		array = np.reshape(array, (-1,4))

		pclFileContent = '# .PCD v.7 - Point Cloud Data file format\nVERSION .7\nFIELDS x y z rgb\n\
		SIZE 4 4 4 4\nTYPE F F F U\nCOUNT 1 1 1 1\nWIDTH %d\nHEIGHT 1\nVIEWPOINT %f %f %f %f %f %f %f \n\
		POINTS %d \nDATA ascii\n' % (int(numPoints), posX, posY, posZ, quatW, quatX, quatY, quatZ, int(numPoints))

		useReflectivityNotColor = True
		if useReflectivityNotColor :
			for p in array :
				pclFileContent +=  '%.5f %.5f %.5f %.2f\n' % (p[0], p[1], p[2], p[3]) 
		else:
			for p in array :
				pclFileContent +=  '%.5f %.5f %.5f %d\n' % (p[0], p[1], p[2], (int(p[3]))) 
			
		timeStamp = '%.2f_%.2f'%(timeStart, timeEnd)
		filename = '../SensorData/'+TestContext.testName+'/pcl/'+ str(ID) + '_' + prefix +'_'+str(imageNum) + '_'  +timeStamp + '.pcd'
		fileObject = open(filename, mode='w')
		fileObject.write(pclFileContent)

		print ("PCL  ", imageNum," recieved! - thread ",ID)
		time.sleep(delay)
		if TestContext.testEnded :
			break
"""

def recieveStereoCamera(client, ID, prefix, dataDest:List[ThreadSafeData], data_definition, test_manager:ALSTestManager):
	imageNum = -1
	StartedProcesses:List[ThreadedWriting] = []
	time.sleep(0)
	while True:
		try:
			data = client.read()
		except Exception as e:
			safe_print('an exception occured while reading the next sensor data')
			safe_print(e)
			test_manager.testEnded = True
			test_manager.result = ALSTestManager.ALS_ERROR
			break
		imageNum += 1

		index = 0
		group_sensor_amount, index = ALSFunc.ReadUint32(data, index)
		recieved_images = []
		recieved_metadatas = []
		lastTime = 0
		for i in range(group_sensor_amount):
			sensor_type, index 		= ALSFunc.ReadString(data, index)
			sensor_path, index 		= ALSFunc.ReadString(data, index)
			image, index, image_width, image_height	= ALSFunc.ReadImage_Group(data, index)
			recieved_images.append(image)
			extra_string, index 	= ALSFunc.ReadString(data, index)
			if len(extra_string) > 0:
				#print(" has extra string len:", len(extra_string))
				recieved_metadatas.append(extra_string)
				parsed_string 		 	= json.loads(extra_string)
				if "T" in parsed_string.keys():
					lastTime 				= round(float(parsed_string["T"]),3)
			# else:
			# 	print(" no extra string")

		safe_print(f'Recieved {group_sensor_amount} images - thread {ID} ImgID:{imageNum} timestamp {lastTime}')

		dataDest[0].WriteData(lastTime)
		dataDest[1].WriteData(recieved_images)
		dataDest[2].WriteData(recieved_metadatas)

		if lastTime > test_manager.save_after_time : 
			started = ThreadedWriting(data_definition,imageNum, test_manager.base_path)
			StartedProcesses.append(started)

		#periodically clean the list
		if len(StartedProcesses)>15:
			for i in range(5):
				proc = StartedProcesses.pop(0)
				proc.waitAllThreads()

		time.sleep(0.01)

		if lastTime > test_manager.session_duration:
			safe_print( 'Session Ended')
			test_manager.testEnded = True
			break

		if imageNum > test_manager.safety_net_kill :
			safe_print( 'SAFETY NET KILL')
			test_manager.testEnded = True
			break

		if test_manager.testEnded :
			safe_print( f'Exiting thread {str(ID)}')
			break
	
	#ended -  wait for files writen
	for P in StartedProcesses:
		P.waitAllThreads()

def ReceiveRadar(client, ID, prefix, dataDest:List[ThreadSafeData], data_definition, test_manager:ALSTestManager):
	frameNum = -1
	StartedProcesses:List[ThreadedWriting] = []
	time.sleep(0)
	while True:
		try:
			data = client.read()
		except Exception as e:
			safe_print('an exception occured while reading the next sensor data')
			safe_print(e)
			test_manager.testEnded = True
			test_manager.result = ALSTestManager.ALS_ERROR
			break

		outputlist = []
		readings, sensor_time = radar_datapoint.msg_to_datapoints(data)
		outputlist.append(readings)
		safe_print(f"Received {len(readings)} radar readings")
		dataDest[0].WriteData(outputlist)
		
		if sensor_time > test_manager.save_after_time : 
			started = ThreadedWriting(data_definition, frameNum, test_manager.base_path)
			StartedProcesses.append(started)

		frameNum += 1

		#periodically clean the list
		if len(StartedProcesses)>15:
			for i in range(5):
				proc = StartedProcesses.pop(0)
				proc.waitAllThreads()
		time.sleep(0.01)
		
		if test_manager.testEnded :
			safe_print( f'Exiting thread {str(ID)}')
			break

	#ended -  wait for files writen
	for P in StartedProcesses:
		P.waitAllThreads()

def ReceiveRadarJson(client, ID, prefix, dataDest:List[ThreadSafeData], data_definition, test_manager:ALSTestManager):
	frameNum = -1
	StartedProcesses:List[ThreadedWriting] = []
	time.sleep(0)
	while True:
		try:
			data = client.read()
		except Exception as e:
			safe_print('an exception occured while reading the next sensor data')
			safe_print(e)
			test_manager.testEnded = True
			test_manager.result = ALSTestManager.ALS_ERROR
			break
		jdata = json.loads(data)
		outputlist = []
		readings, sensor_time = radar_datapoint.json_to_datapoints(jdata)
		outputlist.append(readings)
		safe_print(f"Received {len(readings)} radar readings")
		dataDest[0].WriteData(outputlist)
		
		if sensor_time > test_manager.save_after_time : 
			started = ThreadedWriting(data_definition, frameNum, test_manager.base_path)
			StartedProcesses.append(started)

		frameNum += 1

		#periodically clean the list
		if len(StartedProcesses)>15:
			for i in range(5):
				proc = StartedProcesses.pop(0)
				proc.waitAllThreads()
		time.sleep(0.01)
		
		if test_manager.testEnded :
			safe_print( f'Exiting thread {str(ID)}')
			break

	#ended -  wait for files writen
	for P in StartedProcesses:
		P.waitAllThreads()

		
def CreateBaseDirector(base_directory, test_id):
	TestFolderPath = ospath.join(base_directory,test_id)
	try :
		if not ospath.exists(base_directory):
			osmkdir(base_directory)
		if ospath.exists(TestFolderPath):
			rmtree(TestFolderPath, ignore_errors=True)
			time.sleep(0.1)
		if not ospath.exists(TestFolderPath):
			osmkdir(TestFolderPath)
			osmkdir(TestFolderPath+'/Generation')
			# osmkdir(TestFolderPath+'/Videos')
	except  Exception as e:
		safe_print(e)
	return TestFolderPath

class radar_datapoint():
	def __init__(self, velocity, azimuth, altitude, depth, intensity = 1.0):
		self.velocity = velocity
		self.azimuth = azimuth
		self.altitude = altitude
		self.depth = depth
		self.intensity = intensity

	@staticmethod    
	def msg_to_datapoints(msg):
		import struct
		# print (msg)
		data_points = []
		point_amount = struct.unpack('<i', msg[0:4])[0] *4
		time = struct.unpack('f', msg[-4:])[0]
		data = msg[4:-4]
		array = np.frombuffer(data, dtype=np.dtype("float32"))
		array = np.reshape(array, (-1,5))
		for point in array:
			dp = radar_datapoint(point[0],point[1],point[2],point[3],point[4])
			data_points.append(dp)
		return data_points, time

	@staticmethod    
	def json_to_datapoints(msg):
		data_points = []
		time = msg.get("time")[0]
		data = msg.get("data")
		for point in data:
			dp = radar_datapoint(point[0],point[1],point[2],point[3],point[4])
			data_points.append(dp)
		return data_points, time

	def serialize(self):
		str = "Velocity:{}, Azimuth:{}, Altitude:{}, Depth:{}, Intensity:{}\n".format(self.velocity,self.azimuth,self.altitude,self.depth,self.intensity)
		return str

#=========================================================================================================
# Newer dataset collection utilities, these are stand-alone and not dependent on the code above
#=========================================================================================================

BASE_DIR = ALSFunc.get_sensordata_path()
FSIZE = 4
LID_PT_N = 4 # Number of fields in a lidar data point
LID_SIZE_H = 11 * FSIZE # Lidar data header size
LID_SIZE_D = LID_PT_N * FSIZE # Lidar data point size
OUT_DIR_DEFAULT = "UnnamedDataset"

IMAGE_SAVE_MODE = 1
SAVE_MODE = 2
SAVE_PERF_STATS = False
ALSCLIENT_TIMEOUT = 20

class InvalidOptionsException(Exception):
	pass

# Specializations for different kinds of data are defined in subclasses
class SensorData(ABC):
	sensor_type_str = "UNDEFINED_SENSOR_TYPE"
	def __init__(self):
		self.data = None
		self.meta = None
		self.addl_meta:dict = None
		self.loaded_buffer_size = 0 # used for offset into a sensor group buffer
		self.ingroup = False
		self.timestamp = 0.0
		self.output_path = str()
		self.save_meta = False

	@abstractmethod
	def load(self, buffer, offset=0):
		raise Exception("Method not implemented in subclass ({})".format(inspect.stack()[0][3]))

	@abstractmethod
	def save(self, dir, filename):
		raise Exception("Method not implemented in subclass ({})".format(inspect.stack()[0][3]))

	def valid(self):
			return bool(self.data) and self.loaded_buffer_size > 0
	
	def set(self, new_data, bytes_loaded_from_buffer):
		self.data = new_data
		self.loaded_buffer_size = bytes_loaded_from_buffer

	def overwrite_check(self, full_path, warn=True):
		if ospath.isfile(full_path):
			return False
		else: return True

	def save_metadata(self, filepath):
		if self.meta and self.save_meta:
			try:
				json_data = {}
				if isinstance(self.meta, dict):
					json_data['received_metadata'] = self.meta
				if self.addl_meta and isinstance(self.addl_meta, dict):
					json_data['additional_metadata'] = self.addl_meta
				with open(filepath, "w", encoding="utf-8") as f:
					json.dump(json_data, f, indent=4)
			except Exception:
				print("Failed to save metadata")

	def load_dummy_data(self, index=0):
		raise Exception("Method not implemented in subclass ({})".format(inspect.stack()[0][3]))


class CameraData(SensorData):
	sensor_type_str = "CAM"
	def __init__(self):
		SensorData.__init__(self)

	def load(self, buffer, offset=0):
		i = offset
		# load image
		if not self.ingroup:
			image, i, w, h = ALSFunc.ReadImage_Stream(buffer, i)
		else:
			image, i, w, h = ALSFunc.ReadImage_Group(buffer, i)

		# load metadata (extra string)
		if i < len(buffer):
			extra_str, i = read_str_from_mem_view(buffer, i)
			self.meta = json_from_string(extra_str)
			if self.meta:
				if "T" in self.meta.keys():
					self.timestamp = float(self.meta['T'])
			else:
				self.timestamp = 9999.9999

			b, g, r, a = Image.fromarray((image).astype(np.uint8)).split()
			image = Image.merge("RGB", (r, g, b))

		self.set(image, i - offset)

	def save(self, dir, filename):
		full_path = ospath.join(dir, filename + ".png")

		duplicate_i = 0
		while ospath.isfile(full_path):
			if duplicate_i > 100:
				raise Exception("Too many copies of duplicate filename")
			duplicate_i += 1
			full_path = ospath.join(dir, f"{duplicate_i}_{filename}.png")

		filename = ospath.basename(full_path).removesuffix(".png")

		if IMAGE_SAVE_MODE == 0:
			self.data.save(full_path, "png", optimize=False, compress_level=1) # compression level 0-9
		elif IMAGE_SAVE_MODE == 1:
			image_data = np.array(self.data)
			image_data = cv2.cvtColor(image_data, cv2.COLOR_RGB2BGR)
			cv2.imwrite(full_path, image_data, [int(cv2.IMWRITE_PNG_COMPRESSION), 1])
		self.save_metadata(ospath.join(dir, filename + ".json"))
		self.data = None

	def load_dummy_data(self, index):
		colors = [(78, 184, 230), (25, 112, 148), (24, 83, 107), (20, 64, 82), (20, 44, 54)]
		data = Image.new("RGB", (1920, 1080), colors[index % len(colors)])
		self.set(data, 1)
		


class DummyCameraData(SensorData):
	sensor_type_str = "CAM"
	def __init__(self):
		SensorData.__init__(self)
	# This is used to test loading of an image, but not save it
	def load(self, buffer, offset=0):
		i = offset
		if not self.ingroup:
			size, i, w, h = self.ReadImage_DUMMY(buffer, i)
			if i+FSIZE < len(buffer):
				extra_str, i = ALSFunc.ReadString(buffer, i)
				j = json.loads(extra_str)
				if "T" in j.keys():
					self.timestamp = float(j['T'])
		else:
			raise Exception("debug data class - cannot be used with group")

		self.set(str("dummy"), i - offset)
	
	def ReadImage_DUMMY(self, buffer, index):
		image_height, 	index	= ALSFunc.ReadUint32(buffer, index)
		image_width	, 	index	= ALSFunc.ReadUint32(buffer, index)
		image_channels, index 	= ALSFunc.ReadUint32(buffer, index)
		image_size 	= image_height * image_width * image_channels
		index += image_size
		return image_size, index, image_width, image_height

	def save(self, dir, filename):
		self.data = None
		fullpath = ospath.join(dir, str(self.timestamp)+".txt")
		with open(fullpath, "w") as f:
			f.write("")

class LidarData(SensorData):
	sensor_type_str = "LID"
	def __init__(self):
		SensorData.__init__(self)

	def load(self, buffer, offset=0):
		# Read header
		posX, posY, posZ, quatW, quatX, quatY, quatZ, num_points, tStart, tEnd,\
		num_rays = struct.unpack('<fffffffffff', buffer[offset:offset+LID_SIZE_H])
		# Read data
		self.timestamp = tStart
		data_size = int(num_points * LID_SIZE_D)
		point_array = np.frombuffer(buffer[offset+LID_SIZE_H:offset+LID_SIZE_H+data_size], dtype=np.dtype("float32"))
		point_array = np.reshape(point_array, (-1, LID_PT_N))

		# Serialize to Point Cloud Data format (.pcd compatible with cloud compare)
		pcl = '# .PCD v.7 - Point Cloud Data file format\nVERSION .7\nFIELDS x y z rgb\n\
		SIZE 4 4 4 4\nTYPE F F F U\nCOUNT 1 1 1 1\nWIDTH %d\nHEIGHT 1\nVIEWPOINT %f %f %f %f %f %f %f \n\
		POINTS %d \nDATA ascii\n' % (int(num_points), posX, posY, posZ, quatW, quatX, quatY, quatZ, int(num_points))
		for p in point_array:
			intensity = 1000
			if not math.isinf(p[3]):
				intensity = int(p[3])
			pcl +=  '%.5f %.5f %.5f %d\n' % (p[0], p[1], p[2], intensity)

		self.set(pcl, LID_SIZE_H + data_size)

	def save(self, dir, filename):
		full_path = ospath.join(dir, filename + ".pcd")
		if self.overwrite_check(full_path):
			with open(full_path, mode='w', encoding='utf-8') as f:
				f.write(self.data)
		self.data = None

class RadarData(SensorData):
	sensor_type_str = "RAD"
	def __init__(self):
		SensorData.__init__(self)

	def load(self, buffer:memoryview, offset=0):
		buffer_slice:memoryview = buffer[offset:]
		data:str = None
		try:
			data = str(buffer_slice, 'utf8')
		except Exception as e:
			data = str(buffer_slice[:min(40,len(buffer_slice))], 'utf8', "ignore")
			data = f"{e}: Failed to decode radar data. Partial data string: '{data}'"
			log_to_file(data)

		data_json = None
		try:
			data_json = json.loads(data)
		except json.decoder.JSONDecodeError:
			data_json = None
		if not data_json:
			self.set(data, len(data.encode('utf-8')))
			return

		for i, p in enumerate(data_json["data"]):
			# [ altitude, azimuth, depth, velocity, intensity ] -> [ velocity, azimuth, altitude, depth, intensity ]
			data_json["data"][i] = [ float(p[3]), float(p[1]), float(p[0]), float(p[2]), float(p[4]) ]

		json_str = json.dumps(data_json)
		self.set(json_str, len(json_str.encode('utf-8')))

	def save(self, dir, filename):
		full_path = ospath.join(dir, filename + ".txt")
		if self.overwrite_check(full_path):
			with open(full_path, mode='w', encoding='utf-8') as f:
				f.write(self.data)
		self.data = None

class TextData(SensorData):
	sensor_type_str = "TEXT_DATA_BASE_CLASS"
	def __init__(self):
		SensorData.__init__(self)

	def load(self, buffer, offset=0):
		reading, new_offset = read_str_from_mem_view(buffer, offset)
		self.set(reading, new_offset - offset)

	def save(self, dir, filename):
		full_path = ospath.join(dir, filename + ".txt")
		if self.overwrite_check(full_path):
			with open(full_path, mode='w', encoding='utf-8') as f:
				f.write(self.data)
		self.data = None

class SpeedometerData(TextData):
	sensor_type_str = "SPE"
	def __init__(self):TextData.__init__(self)
	def load(self, buffer, offset=0):TextData.load(self, buffer, offset)
	def save(self, dir, filename):TextData.save(self, dir, filename)

class LaserData(TextData):
	sensor_type_str = "LAS"
	def __init__(self):TextData.__init__(self)
	def load(self, buffer, offset=0):TextData.load(self, buffer, offset)
	def save(self, dir, filename):TextData.save(self, dir, filename)

class GNSSData(TextData):
	sensor_type_str = "GNS"
	def __init__(self):TextData.__init__(self)
	def load(self, buffer, offset=0):TextData.load(self, buffer, offset)
	def save(self, dir, filename):TextData.save(self, dir, filename)

class SplineData(TextData):
	sensor_type_str = "SPL"
	def __init__(self):TextData.__init__(self)
	def load(self, buffer, offset=0):TextData.load(self, buffer, offset)
	def save(self, dir, filename):TextData.save(self, dir, filename)

class IMUData(TextData):
	sensor_type_str = "IMU"
	def __init__(self):TextData.__init__(self)
	def load(self, buffer, offset=0):TextData.load(self, buffer, offset)
	def save(self, dir, filename):TextData.save(self, dir, filename)

class FilteredObjectGetterData(TextData):
	sensor_type_str = "FOG"
	def __init__(self):TextData.__init__(self)
	def load(self, buffer, offset=0):TextData.load(self, buffer, offset)
	def save(self, dir, filename):TextData.save(self, dir, filename)

class AISData(TextData):
	sensor_type_str = "AIS"
	def __init__(self):TextData.__init__(self)
	def load(self, buffer, offset=0):TextData.load(self, buffer, offset)
	def save(self, dir, filename):TextData.save(self, dir, filename)

#=========================================================================================================

class SensorDataClasses():
	def __init__(self):
		# Default classes, can be changed by the user (per-instance)
		self.camera = CameraData
		self.lidar = LidarData
		self.radar = RadarData
		self.speedometer = SpeedometerData
		self.laser = LaserData
		self.gnss = GNSSData
		self.spline = SplineData
		self.imu = IMUData
		self.fog = FilteredObjectGetterData
		self.ais = AISData

	def select_class(self, a, b):
		if a == "CAM" or b == "CAM":
			return self.camera, "CAM"
		elif a == "LID" or b == "LID":
			return self.lidar, "LID"
		elif a == "RAD" or b == "RAD":
			return self.radar, "RAD"
		elif a == "SPE" or b == "SPE":
			return self.speedometer, "SPE"
		elif a == "LAS" or b == "LAS":
			return self.laser, "LAS"
		elif a == "GNS" or b == "GNS":
			return self.gnss, "GNS"
		elif a == "SPL" or b == "SPL":
			return self.spline, "SPL"
		elif a == "IMU" or b == "IMU":
			return self.imu, "IMU"
		elif a == "FOG" or b == "FOG":
			return self.fog, "FOG"
		elif a == "AIS" or b == "AIS":
			return self.ais, "AIS"
		else:
			raise Exception("Error: Found no data class for sensor type {} / {}".format(a,b))

def read_socket_safe(socket):
	if socket._socket is None:
		return None
	header = read_bytes(socket, 4)
	if not header:
		return None
	length = struct.unpack('<L', header)[0]
	return read_bytes(socket, length)

def read_bytes(socket, length):
	buf = bytes()
	while length > 0:
		try:
			data = socket._socket.recv(length)
		except Exception:
			return None
		if not data:
			return None
		buf += data
		length -= len(data)
	return buf

def read_size_from_header(socket):
	if not socket:
		return 0
	header = read_bytes(socket, 4)
	if not header:
		return 0
	return struct.unpack('<L', header)[0]

def view_to_shared(view:memoryview, size:int, shared_mem:shared_memory.SharedMemory, offset = 0):
	shared_mem.buf[offset:offset+size] = view
	return size

def json_from_string(extra_str):
	if not extra_str:
		return None
	try:
		j = json.loads(extra_str)
	except Exception:
		return None
	return j

def save_to_file(data:SensorData, format_str:str, group_name_str:str, alias_str:str, type_str:str, fidx_str:str, g_path_str=str(), g_number_str=str()):
	# legacy format s: (sensor_type=data.sensor_type_str, batch_number=self.data_info.i)
	# legacy format g: (sensor_type=sensor_type,batch_number=self.data_info.i, sensor_path=sensor_path,sensor_number=sensor_number)
	filename = format_str.format(group_name=group_name_str, alias=alias_str, sensor_type=type_str, frame_id=fidx_str,\
					time=data.timestamp, sensor_path=g_path_str, sensor_number=g_number_str)
	if not data.valid():
		raise Exception("Empty data object: {}".format(type(data).__name__))
	path = data.output_path
	if path:
		path = ospath.join(BASE_DIR, path)
		if not ospath.isdir(path):
			osmakedirs(path, exist_ok=True)
	data.save(path, filename)

def log_to_file(message:str):
	time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
	log_row = f"{time_str} {message}\n"
	log_path = ospath.join(BASE_DIR, "DataCollectionLog.txt")
	try:
		with open(log_path, 'a+') as logfile:
			logfile.write(log_row)
	except Exception:
		log_path = ospath.join(ospath.abspath(ospath_sep), "DataCollectionLog.txt")
		with open(log_path, 'a+') as logfile:
			logfile.write(log_row)

#=========================================================================================================

class SubAlloc():
	def __init__(self, offset:int, size:int):
		self.offset = int(offset)
		self.size = size
		self.free = True

class ShmBlock():
	def __init__(self, mem_name, size:int, alloc_num:int, size_margin:float):
		self.name = mem_name
		self.allocs: list[SubAlloc] = []
		suballoc_size = size + round(size_margin * size)
		for i in range(alloc_num):
			self.allocs.append(SubAlloc(suballoc_size * i, suballoc_size))
		self.size = alloc_num * suballoc_size
		self.memory = shared_memory.SharedMemory(name=mem_name, size=self.size, create=True)


	def reserve(self, size, debug_id):
		for alloc in self.allocs:
			if alloc.free and alloc.size >= size:
				alloc.free = False
				return alloc.offset, 0
		return None, 0

	def load_from_view(self, view:memoryview, size:int, offset):
		self.memory.buf[offset:offset+size] = view
		return size
	
	def free(self, offset):
		for i, alloc in enumerate(self.allocs):
			if alloc.offset == int(offset):
				self.allocs[i].free = True
				return True
		return False
	
	def allocs_as_string(self):
		s = " allocs: "
		for a in self.allocs:
			u = "free"
			if not a.free: u = "used"
			s += f"({a.offset},{u}), "
		return s
	
	def percentage_free(self):
		free_num = 0
		for a in self.allocs:
			if a.free:
				free_num += 1
		return (free_num / len(self.allocs)) * 100
			

def shared_memory_get(name):
	try:
		return shared_memory.SharedMemory(name=name, create=False), str()
	except Exception as e:
		return None, str(e)

def discard_read(client:TCPClient2.TCPClient):
	view = client.read_view()
	if not view:
		return False
	if not client.free_view(view):
		raise Exception("Failed to free memory during discard")
	return True

def reserve(mem:ShmBlock, size:int, debug_id="no_id"):
	offset = None
	tries = 0
	while offset == None and tries < 1:
		offset = mem.reliable_reserve(size, debug_id)
		tries += 1
	return offset

class SensorOptions():
	def __init__(self):
		self.additional_metadata:dict = None #Add metadata to saved metadata
		self.group_name = str()
		self.alias = [] # Sensor alias
		self.alloc_num = 16 # Number of allocations made for the buffer, set to a size that will fit to the memory.
		self.batch_limit = 60 #If we are collecting the samples in batches, this should be set.
		self.burn_in_samples = 0 # Number of samples to discard at start
		self.clear_dir = True # Delete existing files at start
		self.data_class_mappings = SensorDataClasses() # Allows overriding sensor types with a custom class
		self.max_variation_sample_size = .1
		self.max_time = 0 # Collection time limit
		self.out_dir = OUT_DIR_DEFAULT # Output directory name
		self.pause_after_collection = True #Pause the manager after collecting batch limit is reached
		self.save_metadata = True # Whether to create metadata files
		self.timeout = 30 # Maximum allowed time between data transmissions

class QueueMessageType:
	Paused = "paused"
	ReportDone = "done"
	# queue type 1
	DataInfo = "data"
	# queue type 2
	FreeMemory = "free"
	StopCommand = "stop"
	ChangeOptions = "recvopt"
	Unpause = "unpause"
	SaveProcessError = "save-process-error"
	Terminated = "terminated"

def receive_queue_message(queue:Queue):
	msg = None
	try:
		msg = queue.get_nowait()
	except QueueEmpty:
		pass

	if type(msg) == DataInfo:
		return QueueMessageType.DataInfo, msg # str, DataInfo
	if type(msg) == SensorOptions:
		return QueueMessageType.ChangeOptions, msg # str, SensorOptions

	elif type(msg) == str:
		if msg == QueueMessageType.Unpause:
			return QueueMessageType.Unpause, 1

		j = json.loads(msg)
		return j["mtype"], j # str, json
	else:
		return None, None

def send_queue_message(queue:Queue, message_type:QueueMessageType, values_dictionary:dict):
	values_dictionary["mtype"] = message_type
	queue.put(json.dumps(values_dictionary))

def empty_function(x): pass

def sanitize_sensor_options(options:SensorOptions):
	options.burn_in_samples = int(options.burn_in_samples)
	options.max_time = float(options.max_time)
	options.max_variation_sample_size = float(options.max_variation_sample_size)
	options.alloc_num = int(options.alloc_num)
	if not options.alloc_num:
		raise InvalidOptionsException(f"{SensorOptions} alloc_num was not set.")
	options.out_dir = str(options.out_dir)
	options.clear_dir = bool(options.clear_dir)
	options.save_metadata = bool(options.save_metadata)
	options.timeout = float(options.timeout)
	if type(options.alias) != list:
		raise InvalidOptionsException(f"{SensorOptions} {options.alias.__name__} was not a list.")
	for alias in options.alias:
		if not isinstance(alias, str):
			if alias != '':
				alias = str(alias)
	if not isinstance(options.data_class_mappings, SensorDataClasses):
		raise InvalidOptionsException(
			f"{type(options.data_class_mappings)} does not inherit from {SensorDataClasses}")
	options.batch_limit = int(options.batch_limit)
	if options.batch_limit == 0 and options.max_time == 0:
		raise InvalidOptionsException("Both batch_limit and max_time cannot be 0")
	options.pause_after_collection = bool(options.pause_after_collection)
	if options.additional_metadata is not None and not isinstance(options.additional_metadata, dict):
		raise InvalidOptionsException(
			f"{options.additional_metadata} is not valid dictionary object."
		)
	return options

def receive_process(addr:str, port:str, queue_in:Queue, queue_out:Queue, data_class:type, filename:str,\
					 options_in:SensorOptions, termination_queue:Queue, log_queue:Queue, mt_rep_q:Queue, \
					main_thread_receive_queue:Queue):
	name = f"{port}_{int(str().join( [str(random.randint(1,9)) for digit in [0] * 8] ))}"
	error:str = None
	opt = copy.copy(options_in)

	try:
		opt = sanitize_sensor_options(opt)
		client = TCPClient2.TCPClient(addr, port, 5)
		client.connect(2)
		client.set_timeouts(-1, opt.timeout, opt.timeout, opt.timeout)
	except InvalidOptionsException as e:
		error = f"Invalid options:\n {e}"
		log(log_queue, f"Invalid options on receive process port {name}")
		send_queue_message(queue_out, QueueMessageType.ReportDone, {"id": f"{name}"})
		msg_type = None
		while msg_type != QueueMessageType.ReportDone:
			msg_type, m = receive_queue_message(queue_in)
		send_queue_message(termination_queue, QueueMessageType.Terminated, {"thread_id": name, "status": "failure", "error": error})
		return
	except Exception as e:
		error = f"Exception at receive process port {name}:\n {e}"
		log(log_queue, f"Exception on receive process port {name}")
		send_queue_message(queue_out, QueueMessageType.ReportDone, {"id": f"{name}"})
		msg_type = None
		while msg_type != QueueMessageType.ReportDone:
			msg_type, m = receive_queue_message(queue_in)
		send_queue_message(termination_queue, QueueMessageType.Terminated, {"thread_id": name, "status": "failure", "error": error})
		return

	shared_mem: ShmBlock = None

	discarded = 0
	batch_counter = 0
	total_sample_amount = 0
	stop = False
	first_recv_time = time.time()
	log(log_queue, f"Receive started on port {name} (using acceleration module)")
	while client.connected() and not stop:
		stop_receive_immediate = False
		while True:
			msg_type, q_msg = receive_queue_message(queue_in)
			if q_msg == None:
				break
			if msg_type == QueueMessageType.SaveProcessError:
				error = q_msg
				stop_receive_immediate = True
				break
			if msg_type == QueueMessageType.StopCommand:
				stop_receive_immediate = True
				log(log_queue,"Stop command received")
			if msg_type == QueueMessageType.ChangeOptions:
				opt = copy.copy(q_msg)
				try:
					opt = sanitize_sensor_options(opt)
				except InvalidOptionsException as e:
					error = f"Invalid options:\n {e}"
					break
				log(log_queue, "Changed sensor options at runtime")
			if msg_type == QueueMessageType.FreeMemory:
				if not shared_mem.free(q_msg["offset"]):
					log_to_file("Failed to free memory")
					log(log_queue, "Failed to free memory at" + str(q_msg["offset"]) + shared_mem.allocs_as_string())
			if msg_type == QueueMessageType.Unpause and batch_counter == -1:
				batch_counter = 0
				log(log_queue, "Receiver resumed")
				break

		if stop_receive_immediate:
			break

		if opt.max_time and time.time() - first_recv_time > opt.max_time:
			log(log_queue, f"Time limit reached ({opt.max_time})")
			break

		if opt.batch_limit != 0 and batch_counter >= opt.batch_limit and batch_counter != -1:
			if opt.pause_after_collection is True:
				batch_counter = -1
				log(log_queue, f"Samples collected, receiving paused for port {name}")
				send_queue_message(main_thread_receive_queue, QueueMessageType.Paused, {"id": f"{name}"})
			else:
				log(log_queue, f"Samples collected, stopping receiver for port {name}")
				break

		if (opt.burn_in_samples and discarded < opt.burn_in_samples) or batch_counter == -1:
			try:
				if discard_read(client):
					discarded += 1
					if batch_counter != -1:
						log(log_queue, f"Burned in samples {discarded}/{opt.burn_in_samples}")
			except Exception as e:
				error = f"Connection failed: {str(e)}"
				break
			continue

		try:
			view = client.read_view()
		except Exception as e:
			error = f"Connection failed: {str(e)}"
			break
		if (not view) or (len(view) < 1):
			continue
		size = len(view)

		if shared_mem is None:
			try:
				shared_mem = ShmBlock(name, size, opt.alloc_num, opt.max_variation_sample_size)
			except Exception as e:
				error = f"Couldn't allocate memory: {str(e)}"
				break

		offset, mem_i = shared_mem.reserve(size, total_sample_amount)
		if offset is None:
			log_to_file("Ran out of allocated space")
			error = "Ran out of allocated space"
			break

		shared_mem.load_from_view(view, size, offset)

		try:
			client.free_view(view)
		except Exception as e:
			error = f"Couldn't free the view {view}"
			break
		is_group = not data_class
		queue_out.put(DataInfo(shared_mem.name, (offset, mem_i), \
						 size, data_class, filename, opt, is_group, total_sample_amount, opt.additional_metadata))
		mt_rep_q.put(f"\"rec\":{size}")
		total_sample_amount += 1
		batch_counter += 1
		log(log_queue, f"Received data on port {name} {batch_counter}/{opt.batch_limit}")
		if not first_recv_time:
			first_recv_time = time.time()
		if total_sample_amount % 2 == 0:
			log(log_queue, f"Allocated memory for port {name} {shared_mem.percentage_free()}% free")

	client.disconnect()
	mem_log_str = ""
	log(log_queue,f"Received {total_sample_amount} on {addr}-{port}{mem_log_str}")
	send_queue_message(queue_out, QueueMessageType.ReportDone, {"id": f"{name}"})

	# wait for saving thread, needed to keep shared memory object alive
	msg_type = None
	while msg_type != QueueMessageType.ReportDone:
		msg_type, m = receive_queue_message(queue_in)
	if error is None:
		send_queue_message(termination_queue, QueueMessageType.Terminated, {"thread_id": name, "status": "success"})
	else:
		send_queue_message(termination_queue, QueueMessageType.Terminated, {"thread_id": name, "status": "failure", "error": error})

def load_override_smart(override, alsclient:ALSClient.Client):
	config_file_type = ""
	config_file_name = ""
	override_text = ""
	if isinstance(override, str):
		# using raw override string
		override_text = override.replace('\\','/')
		override_path = override_text.split(";")[0]
		config_file_type = override_path.rpartition("/")[0] .lower()
		config_file_name = override_path.split("/")[-1].split(".")[0]

	elif isinstance(override, tuple):
		# using ("type", "filename", "override") tuple
		config_file_type = override[0].strip().lower()
		config_file_name = override[1]
		override_text = override[2]
	else:
		return False

	a = config_file_name.strip()
	b = override_text.strip()

	if "weather" in config_file_type:
		alsclient.request_load_weather_with_overrides(a,b, ALSCLIENT_TIMEOUT)
	elif "situation" in config_file_type:
		alsclient.request_load_situation_with_overrides(a,b, ALSCLIENT_TIMEOUT)
	elif "seastate" in config_file_type:
		alsclient.request_load_sea_state_with_overrides(a,b, ALSCLIENT_TIMEOUT)
	elif "sensors/camerasettings" in config_file_type:
		alsclient.request(f"ReloadPostProcessWithOverrides {b}", ALSCLIENT_TIMEOUT)
	else:
		return False

	return True


def simulated_receive_process(name:str, queue_in:Queue, queue_out:Queue, data_class:type, filename:str,\
					 			options:SensorOptions, termination_queue:Queue, log_queue:Queue):
	data_rate = 30
	use_time = options.max_time and options.max_time > 0
	if not (use_counter or use_time):
		raise Exception("No time or sample limit set")

	time_elapsed = 0
	send_index = 0
	num_reported = 0
	while (use_time and time_elapsed < options.max_time):
		msg_type, msg = receive_queue_message(queue_in)

		if msg_type == QueueMessageType.StopCommand:
			log(log_queue, "Simulated receiver stopping")
			break
		if msg_type == QueueMessageType.FreeMemory:
			num_reported += 1

		queue_out.put(DataInfo(mem_name=str(), offset=-1, size=0, data_class=data_class, filename=filename,\
							options=options, is_group_data=False, i=send_index, addl_meta=f"{1/data_rate}"))

		t = 1 / data_rate
		time.sleep(t)
		time_elapsed += t
		send_index += 1

	log(log_queue, f"Dispatched {send_index} (rec{name})")
	send_queue_message(queue_out, QueueMessageType.ReportDone, {"id": f"{name}"})
	msg_type = None
	while msg_type != QueueMessageType.ReportDone:
		msg_type, m = receive_queue_message(queue_in)
	termination_queue.put(f"\"terminated\":\"{name}\"")


class DataInfo():
	def __init__(self, mem_name:str, offset:int, size:int, data_class, filename:str, options:SensorOptions, is_group_data:bool, i, addl_meta:dict):
		self.mem_name = mem_name
		self.offset = offset
		self.size = size
		self.data_class: type[SensorData] = data_class
		self.filename_format_str = filename
		self.is_group_data = is_group_data
		self.i = i
		self.addl_meta = addl_meta
		self.output_path = options.out_dir
		self.save_meta = options.save_metadata
		self.group_name = options.group_name
		self.alias = options.alias
		self.data_class_mappings: SensorDataClasses = options.data_class_mappings

def get_data_info(queue):
	others = []
	infos = []
	while True:
		try:
			msg = queue.get_nowait()
			if type(msg) == DataInfo:
				infos.append(msg)
			else:
				others.append(msg)
		except QueueEmpty:
			break
	for m in others:
		queue.put(m)
	return infos

def save_process(queue_in:Queue, queue_out:Queue, thread_id, log_queue):
	save_process_error_queue=Queue()
	start_time = None
	proc_start_time = time.time()
	error = None

	if SAVE_MODE == 0:
		stop = False
		latest_data_info = None
		while not stop:
			msg_type, q_msg = receive_queue_message(queue_in)
			stop = (msg_type == QueueMessageType.ReportDone)
			if msg_type == QueueMessageType.DataInfo:
				if start_time == None:
					start_time = time.time() # performance test
				singlethreaded_save(q_msg, queue_out)
				latest_data_info = q_msg

	elif SAVE_MODE == 1:
		with Pool(processes=4) as pool:
			stop = False
			jobs = []
			while not stop:
				msg_type, q_msg = receive_queue_message(queue_in)
				stop = (msg_type == QueueMessageType.ReportDone)
				if msg_type == QueueMessageType.DataInfo:
					if start_time == None:
						start_time = time.time() # performance test
					jobs.append(pool.apply_async(singlethreaded_save, (q_msg,None,)))
					offset = jobs[-1].get(None)
					if offset:
						send_queue_message(queue_out, QueueMessageType.FreeMemory,\
						 					 {"offset":f"{offset[0]}", "mem_index":f"{offset[1]}"})

			for j in jobs:
				j.wait(500)

	elif SAVE_MODE == 2:
		stop = False
		latest_data_info = None
		jobs: list[MultiprocessSave] = []
		while not stop:
			msg_type, q_msg = receive_queue_message(save_process_error_queue)
			if msg_type == QueueMessageType.SaveProcessError:
				stop = True
				error = q_msg

			if stop == False:
				msg_type, q_msg = receive_queue_message(queue_in)
				stop = (msg_type == QueueMessageType.ReportDone)
				if msg_type == QueueMessageType.DataInfo:
					if start_time == None:
						start_time = time.time() # performance test
					jobs.append(MultiprocessSave(q_msg, queue_out, save_process_error_queue))
					jobs[-1].start()
					latest_data_info = q_msg

			if len(jobs) > 250:
				jobs[:] = (j for j in jobs if (not j.is_done()))

		pending = 1
		pending_max_wait_time = 50
		wait_start = time.time()
		logged_pending = -1
		while pending > 0:
			pending = 0
			for j in jobs:
				if not j.is_done():
					pending += 1
			if pending > 0:
				if logged_pending != pending:
					log(log_queue, f"Waiting for {pending} save operations ({len(jobs)-pending}/{len(jobs)} finished)")
					logged_pending = pending
				time.sleep(0.1) # this is required, otherwise the done-flags cannot be changed by other threads, maybe GIL-related
				if (time.time() - wait_start) > pending_max_wait_time:
					log(log_queue, f"Exiting save thread after {pending_max_wait_time}s")
					log_to_file("Timed out waiting for saving threads to finish")

	if start_time != None:
		save_debug_str = f"Saving took {time.time() - start_time} s (since start {time.time() - proc_start_time} s) (using acceleration module)"
	else:
		save_debug_str = f"Saving took 0 s (since start {time.time() - proc_start_time} s) (using acceleration module) Nothing saved"
	log(log_queue, save_debug_str)
	if error:
		send_queue_message(queue_out, QueueMessageType.SaveProcessError, error)

	send_queue_message(queue_out, QueueMessageType.ReportDone, {"id": f"{thread_id}"})
	if latest_data_info and latest_data_info.output_path:
		stats_dir = latest_data_info.output_path
		if stats_dir:
			stats_dir = ospath.join(BASE_DIR, stats_dir)
			if SAVE_PERF_STATS and ospath.isdir(stats_dir):
				with open(ospath.join(stats_dir,f"stats_{thread_id}.txt"), "x", encoding="utf-8") as file:
					file.write(save_debug_str)
	log(log_queue, f"Saving completed ({thread_id})")


def singlethreaded_save(data_info:DataInfo, queue_out:Queue):
	name = data_info.mem_name
	offset = data_info.offset[0]
	mem_index = data_info.offset[1]
	mem, err = shared_memory_get(name)
	for i in range (10):
		mem, err = shared_memory_get(name)
		if mem:
			break
		elif i == 0:
			print(f"ERROR: Shared memory for '{name}' unavailable: {err}")
	if mem:
		if not data_info.is_group_data:
			# save single
			data: SensorData = data_info.data_class()
			data.ingroup = False
			data.output_path = data_info.output_path
			data.save_meta = data_info.save_meta
			data.load(mem.buf[offset:offset+data_info.size].tobytes(), 0)
			alias = ''
			#Use first item of the alias list as alias, when saving a single sensor data.
			if type(data_info.alias) == list and len(data_info.alias) > 0:
				alias = data_info.alias[0]
			save_to_file(data, data_info.filename_format_str, data_info.group_name, alias, \
						data.sensor_type_str, data_info.i)
		else:
			# save group
			num_data, offset = ALSFunc.ReadUint32(mem.buf, offset)
			for i in range(num_data):
				sensor_type, offset = read_str_from_mem_view(mem.buf, offset)
				sensor_path, offset = read_str_from_mem_view(mem.buf, offset)
				data_class, sensor_type = data_info.data_class_mappings.select_class(sensor_type, sensor_path)
				data: SensorData = data_class()
				data.ingroup = True
				data.output_path = data_info.output_path
				sensor_number = ''.join(filter(str.isdigit, sensor_path))
				data.save_meta = data_info.save_meta
				data.load(mem.buf, offset)
				alias = ''
				if type(data_info.alias) == list and len(data_info.alias) - 1 >= i:
					alias = data_info.alias[i]
				save_to_file(data, data_info.filename_format_str, data_info.group_name, alias,\
						sensor_type, data_info.i, sensor_path, sensor_number)
				offset += data.loaded_buffer_size
	# save completed
	if queue_out:
		send_queue_message(queue_out, QueueMessageType.FreeMemory, \
					 {"offset":f"{offset}", "mem":f"{name}", "mem_index":f"{mem_index}"})
	return data_info.offset


class MultiprocessSave(threading.Thread):
	def __init__(self, data_info:DataInfo, free_queue, save_process_error_queue:Queue):
		threading.Thread.__init__(self)
		self.data_info = data_info
		self.free_queue = free_queue
		self.save_process_error_queue = save_process_error_queue
		self.lock = threading.Lock()
		self.lock.acquire(blocking=True)

	def run(self):
		name = self.data_info.mem_name
		offset = self.data_info.offset[0]
		mem_index = self.data_info.offset[1]
		if offset < 0:
			return self.run_simulated()

		mem, error = shared_memory_get(name)
		if not mem:
			print(f"Shared memory for '{name}' unavailable: {error}")
			mem_wait_start = time.time()
			while (time.time() - mem_wait_start) < 5 and (mem == None):
				time.sleep(0.05)
				mem, error = shared_memory_get(name)

		if mem:
			try:
				if not self.data_info.is_group_data:
					self.save_single(mem, offset)
				else:
					self.save_group(mem, offset)
			except Exception as e:
				log_to_file(e)
				send_queue_message(self.save_process_error_queue, QueueMessageType.SaveProcessError, \
				{"exception": f"{traceback.format_exc()}"})
			finally:
				self.lock.release()
		else:
			msg = f"Save failed, shared memory exception {error}"
			log_to_file(msg)
			send_queue_message(self.save_process_error_queue, QueueMessageType.SaveProcessError, \
			{"exception": f"{traceback.format_exc()}"})
			return

		send_queue_message(self.free_queue, QueueMessageType.FreeMemory, \
					  	{"offset": f"{offset}","mem_index":f"{mem_index}", "i":f"{self.data_info.i}", })

	def save_single(self, mem:shared_memory.SharedMemory, offset = 0):
		data: SensorData = self.data_info.data_class()
		data.ingroup = False
		data.output_path = self.data_info.output_path
		data.save_meta = self.data_info.save_meta
		data.addl_meta = self.data_info.addl_meta
		data.load(mem.buf[offset:offset + self.data_info.size].tobytes(), 0)
		alias = ''
		#Use first item of the alias list as alias, when saving a single sensor data.
		if type(self.data_info.alias) == list and len(self.data_info.alias) > 0:
			alias = self.data_info.alias[0]
		save_to_file(data, self.data_info.filename_format_str, self.data_info.group_name, alias,\
					data.sensor_type_str, self.data_info.i)

	def save_group(self, mem:shared_memory.SharedMemory, offset = 0):
		num_data, offset = ALSFunc.ReadUint32(mem.buf, offset)
		for i in range(num_data):
			sensor_type, offset = read_str_from_mem_view(mem.buf, offset)
			sensor_path, offset = read_str_from_mem_view(mem.buf, offset)
			data_class, sensor_type = self.data_info.data_class_mappings.select_class(sensor_type, sensor_path)
			data: SensorData = data_class()
			data.ingroup = True
			data.output_path = self.data_info.output_path
			sensor_number = ''.join(filter(str.isdigit, sensor_path))
			data.save_meta = self.data_info.save_meta
			data.addl_meta = self.data_info.addl_meta

			alias = ''
			if type(self.data_info.alias) == list and len(self.data_info.alias) - 1 >= i:
				alias = self.data_info.alias[i]
			data.load(mem.buf, offset)
			save_to_file(data, self.data_info.filename_format_str, self.data_info.group_name, alias,\
						sensor_type, self.data_info.i, sensor_path, sensor_number)
			offset += data.loaded_buffer_size

	def run_simulated(self):
		data: SensorData = self.data_info.data_class()
		data.ingroup = False
		data.save_meta = False
		data.output_path = self.data_info.output_path
		data.load_dummy_data(index=self.data_info.i)
		save_to_file(data, self.data_info.filename_format_str, self.data_info.group_name, self.data_info.alias,\
					data.sensor_type_str, self.data_info.i)
		send_queue_message(self.free_queue, QueueMessageType.FreeMemory, {"offset":"-1", "i":"-1", "mem_index":"-1"})
		self.lock.release()

	def is_done(self):
		if self.lock.acquire(blocking=False):
			self.lock.release()
			return True
		else:
			return False
		


def read_str_from_mem_view(buffer, offset):
	# same as ALSHelperFunctionLibrary.ReadString but this one works with memory views
	str_val_len, offset = ALSFunc.ReadUint32(buffer, offset)
	str_val_view = buffer[offset:offset + str_val_len]
	str_val = str()
	try:
		str_val = str(str_val_view, 'utf8')
	except Exception:
		err_str = "error string could not be generated"
		try:
			err_str = str(buffer[offset:offset+max(str_val_len,40)], 'utf8', "ignore")
		except Exception as exc:
			raise Exception(f"Failed to decode string: '{err_str}'") from exc
		#return (str(), offset)
	return (str_val, offset + str_val_len)


def log_process(log_queue:Queue):
	enabled = False
	stop = False
	while not stop:
		try:
			msg:str = log_queue.get_nowait()
			if "log_enable" in msg:
				enabled = "True" in msg.split(":")[1]
			elif "stop" == msg:
				if enabled and "log" in msg:
					while log_queue.empty() is False:
						msg:str = log_queue.get_nowait()
						print(msg.split(":"))[1]
				stop = True
			elif enabled and "log" in msg:
				print(msg.split(":")[1])
		except QueueEmpty:
			continue

def log(queue, message):
	queue.put("log:{}".format(message.replace(":",";")))

#=========================================================================================================
class ManagerState(Enum):
	STOPPED = 1
	RUNNING = 2
	PAUSED = 3

class ProcessId():
	SAVE_PROCESS = "save-process"
	RECEIVE_PROCESS = "receive-process"

class SensorThreadManager():
	def __init__(self):
		self.data_queues = []
		self.done_queues = []
		self.termination_queue = Queue()
		self.terminated_num = 0
		self.paused_num = 0
		self.log_queue = Queue()
		self.log_thread = None
		self.threads = []
		self.thread_ids = []
		self.state = ManagerState.STOPPED
		self.clear_dirs = []
		self.main_thread_report_queue = Queue()
		self.main_thread_receive_queue = Queue()

	def add(self, addr:str, port:int, data_class:Type[SensorData], filename:str, options = SensorOptions()):
		if self.state != ManagerState.STOPPED:
			raise Exception("Please stop the manager before setting up the sensors.")
		if not issubclass(data_class, SensorData):
			raise Exception(f"Data class is doesn't inherit {SensorData} class")
		if not isinstance(options, SensorOptions):
			raise Exception(f"options is not instance of {SensorOptions}")
		self.__add_any(data_class, str(addr), str(port), str(filename), options)

	def add_group(self, addr:str, port:int, filename:str, options = SensorOptions()):
		if self.state != ManagerState.STOPPED:
			raise Exception("Please stop the manager before setting up the sensors.")
		if not isinstance(options, SensorOptions):
			raise Exception(f"options is not instance of {SensorOptions}")
		self.__add_any(None, str(addr), str(port), str(filename), options)


	def __add_any(self, data_class, addr, port, filename, options:SensorOptions):
		options = copy.copy(options)
		self.data_queues.append(Queue())
		self.done_queues.append(Queue())
		id = f"{ProcessId.RECEIVE_PROCESS}-{port}-{len(self.thread_ids)}"
		self.thread_ids.append(id)

		self.threads.append(Process(name=id,\
								target=receive_process,args=(\
								addr,port,self.done_queues[-1],self.data_queues[-1],\
								data_class,filename,options,self.termination_queue,self.log_queue,self.main_thread_report_queue,
								self.main_thread_receive_queue)))

		if options.clear_dir and options.out_dir and not (options.out_dir in self.clear_dirs):
			self.clear_dirs.append(options.out_dir)

	def start(self):
		if self.state == ManagerState.RUNNING:
			return
		if self.state == ManagerState.PAUSED:
			self.log_add("Receiver is currently paused, please unpause to continue collection.")
			return
		if not self.thread_ids: raise Exception("Error: No sensors added")
		self.log_add("Starting")
		self.state = ManagerState.RUNNING

		for i in range(len(self.threads)):
			id = f"{ProcessId.SAVE_PROCESS}-{i}" # save thread id
			self.threads.append(Process(name=id, target=save_process, args=(\
								self.data_queues[i],self.done_queues[i],id,self.log_queue,)))
		self.log_thread = Process(target=log_process, args=(self.log_queue,))
		self.log_thread.start()

		[self.clear_dir(d) for d in self.clear_dirs]
		[t.start() for t in self.threads if isinstance(t, Process)]
		register_manager_instance(self)

	def start_simulate_receive(self, num_sensors:int, options:SensorOptions, filename:str, data_class):
		for i in range(0, num_sensors):
			self.data_queues.append(Queue())
			self.done_queues.append(Queue())
			options_copy = copy.copy(options)
			options_copy.alias = f"SIM{i}"
			self.thread_ids.append(options_copy.alias)
			self.threads.append(Process(target=simulated_receive_process,args=(\
								str(i),self.done_queues[-1],self.data_queues[-1],\
								data_class,filename,options_copy,self.termination_queue,self.log_queue,)))
			if options_copy.clear_dir and not (options_copy.out_dir in self.clear_dirs):
				self.clear_dirs.append(options_copy.out_dir)
		self.start()

	def stop(self, wait=True):
		error = None
		if self.state == ManagerState.STOPPED:
			return
		for q in self.done_queues:
			send_queue_message(q, QueueMessageType.StopCommand, {"id": "main"})

		if wait:
			self.log_add("Waiting for threads...")
			for t in self.threads:
				if isinstance(t, Process):
					t.join(timeout=60)
					if t.is_alive():
						self.log_add(f"Thread {t.name} timed out")
			self.threads.clear()
			self.log_add("Done")
			self.log_queue.put("stop")
			[self.log_thread.join()]
		self.done_queues.clear()
		self.data_queues.clear()
		self.thread_ids.clear()

		#Clear the pause message queue.
		msg_type = QueueMessageType.Paused
		while msg_type == QueueMessageType.Paused:
			msg_type, msg = receive_queue_message(self.main_thread_receive_queue)

		#Clear termination queue
		msg_type = QueueMessageType.Terminated
		while msg_type == QueueMessageType.Terminated:
			msg_type, msg = receive_queue_message(self.termination_queue)
			if msg:
				self.terminated_num += 1
				if isinstance(msg, dict) and "error" in msg:
					error = msg['error']
				elif isinstance(msg, str) and "error" in msg:
					error = msg
		#Clear removable directories from list of clearable directories
		self.clear_dirs.clear()
		#Reset terminated and paused thread amount.
		self.terminated_num = 0
		self.paused_num = 0
		self.state = ManagerState.STOPPED
		if error:
			raise Exception(error)

	def __del__(self):
		if self.state != ManagerState.STOPPED:
			print("{} unexpectedly destroyed, possible data loss".format(type(self).__name__))

	def log_add(self, msg):
		if self.state != ManagerState.STOPPED and self.log_thread:
			log(self.log_queue, msg)
		else:
			print(msg)

	def log_enable(self, enable):
		self.log_queue.put("log_enable:{}".format(str(enable)))

	def unpause(self):
		if self.state != ManagerState.PAUSED:
			return
		#Send unpause message to message queues in subprocesses.
		for q in self.done_queues:
			q.put(QueueMessageType.Unpause)
		self.paused_num = 0
		self.state = ManagerState.RUNNING

	def stop_if_finished(self):
		error = None
		msg_type = QueueMessageType.Terminated
		while msg_type == QueueMessageType.Terminated:
			msg_type, msg = receive_queue_message(self.termination_queue)
			if msg:
				self.terminated_num += 1
				if isinstance(msg, dict) and "error" in msg:
					error = msg['error']
				elif isinstance(msg, str) and "error" in msg:
					error = msg
		if self.terminated_num >= len(self.thread_ids) or error is not None:
			self.thread_ids = [] # prevent sending unnecessary stop commands
			self.stop(wait=True)
		if error is not None:
			raise Exception(error)
		return self.state != ManagerState.STOPPED

	def is_paused(self):
		if self.state == ManagerState.PAUSED:
			return True
		msg_type = QueueMessageType.Paused
		while msg_type == QueueMessageType.Paused:
			msg_type, msg = receive_queue_message(self.main_thread_receive_queue)
			if msg_type == QueueMessageType.Paused:
				self.paused_num += 1
		count = 0
		for thread_id in self.thread_ids:
			if isinstance(thread_id, str) and thread_id.startswith(ProcessId.RECEIVE_PROCESS):
				count += 1
		if self.paused_num >= count:
			self.state = ManagerState.PAUSED
		return self.state == ManagerState.PAUSED

	def receive_message(self, queue:Queue, match_str):
		try:
			msg = queue.get_nowait()
			if type(msg) == str and match_str in msg:
				value = msg.split(match_str)[1]
				if value:
					return value
				else:
					return True
			else:
				queue.put(msg)
		except QueueEmpty: pass
		return False

	def get_data_reports(self):
		new_message = None
		messages = []
		while new_message or new_message == None:
			new_message = self.receive_message(self.main_thread_report_queue, "\"rec\":")
			if new_message:
				messages.append(new_message)
			if len(messages) > 150:
				self.log_add("Large number of data report entries (may stall)")
		return messages

	def clear_dir(self, dir_name):
		path = ospath.join(BASE_DIR, dir_name)
		if ospath.isdir(path):
			self.log_add("Clearing directory {}".format(path))
			for f in listdir(path):
				f = ospath.join(path, f)
				if ospath.isfile(f):
					removefile(f)

	# legacy signature: self, als_client, options=SensorOptions(), timestamps_only=False
	def auto_setup(self, als_client, options=SensorOptions(), filename_format_str=str()):
		if self.state != ManagerState.STOPPED:
			raise Exception("Please stop the manager before setting up the sensors.")
		
		data = json.loads(als_client.request("GetSensorDistributionInfo"))

		groups = data["groups"]
		sensors = data["sensors"]
		used_groups = []
		used_sensors = []
		grouped_sensors = []

		for group in groups:
			group_name = group["group_name"]
			num_in_group = 0
			aliases = []
			for sensor in sensors:
				if sensor["sensor_group"] == group_name:
					num_in_group += 1
					aliases.append(sensor["sensor_alias"])
					grouped_sensors.append(sensor)

			if str(group["stream_to_network"]).strip().lower() == "true":
				used_groups.append((group, num_in_group, aliases))

		for sensor in sensors:
			if not sensor in grouped_sensors and str(sensor["stream_to_network"]).strip().lower() == "true":
				used_sensors.append(sensor)

		auto_dir_name = options.out_dir == OUT_DIR_DEFAULT
		num_groups_skip = len(groups)-len(used_groups)
		if num_groups_skip:
			self.log_add("{} group{} not streaming, skipped".format(num_groups_skip, "s" if num_groups_skip>1 else ""))
		if not used_sensors:
			self.log_add("Warning: No streaming sensors found")

		# Add groups
		for g in used_groups:
			group = g[0]
			num = g[1]
			aliases = g[2]
			group_name = group["group_name"]
			group_addr = group["socket_ip"]
			group_port = group["socket_port"]
			if num < 1:
				# workaround in case simulation does not report sensors, ideally should not be shipped like this
				#self.log_add("No sensors found for group '{}'".format(group_name))
				#continue
				# uncomment above lines to re-enable empty group discarding
				print(f"Group {group_name} force-enabled")
			options_copy = copy.copy(options)
			if auto_dir_name:
				options_copy.out_dir = group_name
			if group_name:
				options_copy.group_name = group_name
			options_copy.alias = aliases
			if not filename_format_str:
				default_format = "{sensor_type}{sensor_number}_{frame_id}"
				default_format = "{alias}_" + default_format if len(options_copy.alias) > 0 and any(alias for alias in options_copy.alias) else default_format
				default_format = "{group_name}_" + default_format if len(group_name) > 0 else default_format
				self.add_group(group_addr, group_port, default_format, options_copy)
			else:
				self.add_group(group_addr, group_port, filename_format_str, options_copy)
			self.log_add("Added group '{g}' ({n} sensors) {addr}:{p}".format(g=group_name, n=num, addr=group_addr, p=group_port))
		
		# Add sensors
		type_counts = {}
		for i, sensor in enumerate(used_sensors):
			type = sensor["sensor_type"]
			if type in type_counts:
				type_counts[type] += 1
			else:
				type_counts[type] = 1

			sensor_id = "{t}{i}".format(t=type, i=type_counts[type]-1)
			data_class = options.data_class_mappings.select_class(type, type)[0]
			options_copy = copy.copy(options)
			if auto_dir_name:
				options_copy.out_dir = sensor_id
			options_copy.alias.append(sensor["sensor_alias"] if len(sensor["sensor_alias"]) > 0 else str())
			if not filename_format_str:
				default_format = "{sensor_type}{sensor_number}_{frame_id}"
				default_format = "{alias}_" + default_format if len(options_copy.alias) > 0 and any(alias for alias in options_copy.alias) else default_format
				self.add(sensor["sensor_ip"], sensor["sensor_port"], data_class, default_format, options_copy)
			else:
				self.add(sensor["sensor_ip"], sensor["sensor_port"], data_class, filename_format_str, options_copy)
			if len(options_copy.alias[i]) > 0:
				self.log_add("Added sensor '{a}' {id} {addr}:{p}".format(a=options_copy.alias[i], id=sensor_id,\
														 addr=sensor["sensor_ip"], p=sensor["sensor_port"]))
			else:
				self.log_add("Added sensor {id} {addr}:{p}".format(a=options_copy.alias[i], id=sensor_id,\
														 addr=sensor["sensor_ip"], p=sensor["sensor_port"]))

	def set_options(self, new_options:SensorOptions, port:int=None):
		if port is None:
			for q in self.done_queues:
				q.put(new_options)
		else:
			for item in self.thread_ids:
				if isinstance(item, str) and str(port) in item:
					start = item.find(str(port)) + 5
					index = int(item[start:])
			self.done_queues[index].put(new_options)

	def collect_samples_until_paused(self, logging_enabled:bool=True, delta_time:float=0.05):
		if self.state == ManagerState.STOPPED:
			self.start()
			self.log_enable(logging_enabled)
		elif self.state == ManagerState.PAUSED:
			self.unpause()
		while self.state == ManagerState.RUNNING:
			time.sleep(delta_time)
			self.is_paused()
			self.stop_if_finished()

DEFAULT_SIGNAL_HANDLER = None
MANAGER_INSTANCES = []

def register_manager_instance(inst):
	global MANAGER_INSTANCES
	if inst in MANAGER_INSTANCES:
		return
	MANAGER_INSTANCES.append(inst)
	register_signal_handler()

def register_signal_handler():
	global DEFAULT_SIGNAL_HANDLER
	if not DEFAULT_SIGNAL_HANDLER:
		DEFAULT_SIGNAL_HANDLER = signal.getsignal(signal.SIGINT)
		signal.signal(signal.SIGINT, handle_signal) # Register custom signal handler

def handle_signal(sig, frame):
	global DEFAULT_SIGNAL_HANDLER, MANAGER_INSTANCES
	signal.signal(signal.SIGINT, DEFAULT_SIGNAL_HANDLER) # Restore default signal handler
	[mgr.stop(True) for mgr in MANAGER_INSTANCES] # Wait for graceful shutdown

#=========================================================================================================
