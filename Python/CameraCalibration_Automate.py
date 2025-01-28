
import sys, os
from os.path import isfile, join
from shutil import rmtree
import ALSLib.ALS_TestDefinitionGenerator, ALSLib.ALSHelperFunctionLibrary
from ALSLib.ALSHelperFunctionLibrary import get_sensordata_path, get_config_path

directories_created = []
tests_to_run = ["CameraCalibration_Scenario"]
variants = [get_config_path("/SituationOverrides/CameraCalibration_Dataset_Overrides.txt")]


for Num, test in enumerate(tests_to_run):
	FolderName = test
	
	for variantID, variant in enumerate(variants):
		scenario, parameters = ALSLib.ALS_TestDefinitionGenerator.get_scenario_parameters_from_file(variant)
		content = ALSLib.ALS_TestDefinitionGenerator.generate_test_description(parameters, scenario)

		for count, line in enumerate(content.splitlines()):
			overrides=line.split()[-1]
			command = "py CameraCalibration_DatasetCollection.py "+test+" \""+overrides+"\""
			print( command)
			os.system( command)
			
			folder_path = get_sensordata_path(str(Num)+"_"+FolderName+"_"+str(variantID)+"_"+str(count))
			if os.path.exists(folder_path):
				rmtree(folder_path)
			os.rename(get_sensordata_path("/test_id_local"), folder_path)

			directories_created.append(folder_path)


print("The following directories were created: ", directories_created)

