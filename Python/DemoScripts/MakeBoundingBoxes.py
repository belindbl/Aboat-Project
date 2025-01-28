import xml.etree.ElementTree as ET
import numpy, cv2, json
from PIL import Image
from ALSLib.ALSHelperFunctionLibrary import get_sensordata_path, get_config_path

def GetColorMap(palette_image:str):
	im = Image.open(palette_image, 'r')
	width, height = im.size
	pixel_values = list(im.getdata())
	pixel_values = numpy.array(pixel_values).reshape((width, height, 3))
	return pixel_values

def GetColorID(color, pixel_values):
	i = 0
	bestI = 0
	bestDist = 999
	for col in pixel_values:
		if (col[0] == color).all():
			return i
		dist = numpy.absolute((col[0] - color)).sum()
		if dist <= 1:
			print("Near miss color: ", color)
			return i
		if dist < bestDist:
			bestI = i
			bestDist = dist
		i+=1
	print("Color Not Found: ", color, " using ", pixel_values[bestI], " (dist:", bestDist,")")
	return bestI

def MapColorPalette(xml_file:str):
	mytree = ET.parse(xml_file)
	myroot = mytree.getroot()

	categories_to_colors = {}
	ordered_categories = [] # array is enough, ids are the key to the category name
	for i in range(256) :
		ordered_categories.append("Unknown")

	for rule in myroot.iter("Rule"):
		colorID = rule.attrib.get('colorID')
		colorRange = rule.attrib.get('colorRange')		
		color_start = color_end = 0
		if colorID:
			color_start = color_end = int(colorID)
		if colorRange:
			color_start = int(colorRange.split(':')[0])
			color_end = int(colorRange.split(':')[1])
		while color_start <= color_end:
			category = rule.attrib.get('category')
			if not category:
				print("error, no categry found in the rule: ",rule)
				continue
			ordered_categories[color_start] = category
			if categories_to_colors.get(category) :
				if color_start not in categories_to_colors[category]: 
					categories_to_colors[category].append(color_start)
			else:
				categories_to_colors[category] = [color_start]
			color_start += 1

	return categories_to_colors, ordered_categories


def GetBoxesFromImage(mask):
	box_list = []
	for color in  numpy.unique(mask.reshape(-1, mask.shape[2]), axis=0 ):
		
		lower = numpy.array(color, dtype="uint8")
		upper = numpy.array(color, dtype="uint8")
		img_mask = cv2.inRange(mask, lower, upper)
		x, y, w, h = cv2.boundingRect(img_mask)

		#color_np = np
		count = numpy.sum(numpy.all(mask == color, axis=2))
		box_list.append((x,y,w,h, color, int(count)))
	return box_list

def GetBoxes(image_gt:str):
	mask = numpy.array(Image.open(image_gt))
	return GetBoxesFromImage(mask)

def ShowImage(image):
	try:
		cv2.imshow("image_boxes",image)
	except Exception as e:
		print("Unable to show image: "+str(e))


def PrintBoxesOnImage(image_src:str, out_image:str, box_list, categories_to_colors, ordered_categories, pixel_array):
	img = cv2.imread(image_src)
	for (x,y,w,h, color, num_pix) in box_list:
		colorID = GetColorID(color, pixel_array)
		category = ordered_categories[colorID]
		if category == "Unknown":
			continue		
		if category == "Sea":
			continue
		if category == "default":
			continue

		pixel_to_area_ratio = (w*h)/num_pix 
		should_discard_box = w < 7 or h < 7 or pixel_to_area_ratio > 3.5 

		colorrec = (int(color[2]), int(color[1]), int(color[0])) #BGR
		if should_discard_box:
			colorrec = (0,0,255)
		img = cv2.rectangle(img, (x, y), (x+w, y+h), colorrec,2 )
		font = cv2.FONT_HERSHEY_SIMPLEX 
		img = cv2.putText(img, "{:s}_{:.1f}".format(category, pixel_to_area_ratio), (x,y-3),font, 0.5, colorrec,1)
	print("!->",out_image)
	cv2.imwrite(out_image,img)
	ShowImage(img)
	cv2.waitKey(1)# & 0xFF # allows opencv to update the current image

def GetBoxesJson(box_list, categories_to_colors, ordered_categories, pixel_array):
	box_text = []
	for (x,y,w,h, color,num_pix) in box_list:
		colorID = GetColorID(color, pixel_array)
		category = ordered_categories[colorID]
		if category == "Unknown":
			continue
		bbox = [x, y, w, h]
		colorrec = (int(color[0]), int(color[1]),int(color[2]))#rgb
		box_str = {"category": category, "bbox":bbox,"color":colorrec, "instance_id":colorID, "num_pixels":num_pix}
		box_text.append(box_str)

	return box_text

def GetCategoriesJson(categories_to_colors):
	categories_text = []
	# i=-1
	# for cat in categories_to_colors:
	# 	i=i+1
	for i, cat in enumerate(categories_to_colors):
		category_str = { "id": i , "name":cat}
		categories_text.append(category_str)
	return categories_text

def GetImageJson(image_file:str):
	image_text = []
	img = Image.open(image_file)
	width, height = img.size
	image_text.append({"file_name":image_file,"width":width, "height": height, "license":"AILiveSim Confidential"})
	return image_text

def SaveMetadataToJson(json_file, image_to_box, image_gt,  xml_file,palette_image:str=get_sensordata_path("/seg_color_pallet.tga")):
	categories_to_colors, ordered_categories = MapColorPalette(xml_file)
	pixel_array = GetColorMap(palette_image)
	boxes_list = GetBoxes(image_gt)
	category_map = GetCategoriesJson(categories_to_colors)

	box_map = GetBoxesJson(boxes_list, categories_to_colors, ordered_categories, pixel_array)
	image_map = GetImageJson(image_to_box)
	metadata = {"images":image_map
			, "categories":category_map
			, "annotations": box_map}
	
	with open(json_file, "w") as f:
		json.dump(metadata, f, indent=4)
	

if __name__ == "__main__":
	palette_image = get_sensordata_path("/seg_color_pallet.tga")
	xml_file = get_config_path("/Sensors/GroundTruthColorMapping.xml")
	# image_gt = "../SensorData/UH_Test1_1/1_Segmt_0_6.80.png"
	# image_to_box = "../SensorData/UH_Test1_1/0_Image_0_6.80.png"
	# json_file= "../SensorData/UH_Test1_1/meta_Image_0_6.80.txt"
	image_gt = get_sensordata_path("/test_id_local/1_Segmt_1_5.80.png")
	image_to_box = get_sensordata_path("/test_id_local/0_Image_1_5.80.png")
	json_file= get_sensordata_path("/test_id_local/meta_Image_1_5.80.txt")

	image_gt = get_sensordata_path("/0_Parking_Overcast/1_Segmt_7_10.80.png")
	image_to_box = get_sensordata_path("/0_Parking_Overcast/0_Image_7_10.80.png")
	json_file= get_sensordata_path("/0_Parking_Overcast/meta_Image_7_10.80.txt")
	categories_to_colors, ordered_categories = MapColorPalette(xml_file)
	boxe=GetBoxes(image_gt)
	print(categories_to_colors)
	print(ordered_categories)
	print(boxe)
	#SaveMetadataToJson(json_file,image_to_box, image_gt, xml_file, palette_image)


	# categories_to_colors, ordered_categories = MapColorPalette(xml_file)
	# pixel_array = GetColorMap(palette_image)
	# boxes_list = GetBoxes(image_gt)
	# PrintBoxesOnImage(image_to_box,"../SensorData/Test.png", boxes_list, categories_to_colors, ordered_categories, pixel_array)

