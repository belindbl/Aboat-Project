#!/usr/bin/python

import sys
import os
import argparse


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


plugin_name      = os.path.basename(sys.argv[0]).replace("uninstall", "").replace(".py", "")
print("\n ### Uninstalling "+plugin_name+" ###\n")

######### validating the variables #########

ap = argparse.ArgumentParser()
ap.add_argument("-c", "--CleanConfigFile", required=False, nargs='?', default=argparse.SUPPRESS, const=False, type=str2bool, help="Clean the default editor subscene from /Config/\".ini\": if not specified it will ask the question, -c will remove, or use -c True or -c False")
args = vars(ap.parse_args())

skip_question = False
want_clean_defaults = False
if "CleanConfigFile" in args:
    skip_question = True
    want_clean_defaults = args['CleanConfigFile']

if not skip_question and os.path.isfile(os.path.join("../","Default_Subscenes_"+plugin_name+".ini")):
    print("The file Default_Subscenes_",plugin_name,".ini contains the subscenes that get loaded by default when the scene is opened in the Editor with \"0\"\nWould you like to remove it?(y/N)>")
    userinput = input()
    want_clean_defaults = userinput.lower() in ["y", "yes","1"]


######### uninstall all files #########

with open(plugin_name + ".files.txt") as filelist:
    files_to_delete   = filelist.readlines()
    files_to_delete   = [x.strip() for x in files_to_delete] 

for file in files_to_delete:
    if not want_clean_defaults and "default_subscenes_" in file.lower() and ".ini" in file.lower():
        print("Skipping: " + file)
        continue    

    if os.path.exists(file):
            os.remove(file)
            print("Deleted " + file)
    
    #restore the ground truth file backup
    if "GroundTruthColorMapping.xml" in file:
        backup_file_name = file + ".bak"
        if os.path.exists(backup_file_name):
            os.rename(backup_file_name, file)


print("\n ### Uninstallation Done ###\n")
