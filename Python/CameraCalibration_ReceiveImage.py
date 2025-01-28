from PIL import Image
import ALSLib.TCPClient, struct, os
import numpy as np
import ALSLib.ALSHelperFunctionLibrary as ALSFunc

###############################################################################
# Very naive implemetation of collecting images form a sensor
# Note that this requires that the sitation is already loaded and running in 
# AILiveSim as it will not send any commands to start a situation.
################################################################################

def saveImage(img):
	im = Image.fromarray(img)
	b, g, r, a = im.split()
	im = Image.merge("RGB", (r, g, b))
	imageNum = images_taken
	filename = '{Path}/img{Num}.png'.format(Path = image_set_path, Num = imageNum)
	print("Saving image {Num}".format(Num = imageNum))
	im.save(filename, 'png' )

HOST = '127.0.0.1'
PORT = 8881

client = ALSLib.TCPClient.TCPClient(HOST, PORT, 5 )
client.connect(5)

image_set_name = "Calibration"
image_set_path = ALSFunc.get_sensordata_path('/images/{Set}/').format(Set = image_set_name)
if not os.path.exists(image_set_path):
	os.makedirs(image_set_path)

image_amount = 15
images_taken = 0

while images_taken < image_amount:  
	data = client.read()    
	index = 0
	img, index, width, height = ALSFunc.ReadImage_Stream(data, index)
	saveImage(img)
	images_taken += 1
	