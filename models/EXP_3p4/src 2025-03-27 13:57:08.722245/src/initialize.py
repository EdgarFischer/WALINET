import os
import sys
import shutil
import numpy as np
import math, copy, time
import random
import argparse
from datetime import datetime


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def read_args(params):
    parser = argparse.ArgumentParser()
    parser.add_argument("--name")
    parser.add_argument("--dropout")
    parser.add_argument("--preload")
    parser.add_argument("--preload_model")

    args = parser.parse_args()

    if args.name:
        params["model_name"] = args.name
        params["path_to_model"] = "/workspace/walinet/models/" + params["model_name"] + "/"
        params["verbose"] = params["model_name"] == "test"
    if args.dropout:
        params["dropout"] = float(args.dropout)
    if args.preload:
        params['preload'] = str2bool(args.preload)
    if args.preload_model:
        params['preload_model'] = args.preload_model

    return params


def initialize_model_folder(params):
    if params["clean_model"] == True:
        if params["model_name"] != "test":
            if os.path.isdir(params["path_to_model"]) == True:
                print("Model already exists. Choose different model name.")
                sys.exit()
            else:
                os.makedirs(params["path_to_model"] + "predictions/")
        else:
            if os.path.isdir(params["path_to_model"]) == True:
                shutil.rmtree(params["path_to_model"])
            os.makedirs(params["path_to_model"] + "predictions/")
    my_copy(params["path_to_model"])
    
    file = open(params["path_to_model"] + "params.txt", "a")
    for key in params.keys():
        file.write(key + ': ' + str(params[key]))
        file.write('\n')
    file.close()



def my_copy(path_to_model):
    path = "src/"
    now = datetime.now()
    dest = path_to_model + "src " + str(now) +"/"

    os.makedirs(dest)
    shutil.copy("srun.sh", dest + "srun.sh")
    shutil.copy("run.py", dest + "run.py")
    shutil.copy("config.py", dest + "config.py")

    dest = dest+path
    os.mkdir(dest)
    src_files = os.listdir(path)
    for file_name in src_files:
        full_file_name = os.path.join(path, file_name)
        if os.path.isfile(full_file_name):
            shutil.copy(full_file_name, dest + file_name)