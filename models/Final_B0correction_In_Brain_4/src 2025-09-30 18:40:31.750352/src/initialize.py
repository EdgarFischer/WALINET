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
    base = params["path_to_model"]
    preds = os.path.join(base, "predictions")

    #––– ensure full perms on creation –––
    old_umask = os.umask(0)
    try:
        if params.get("clean_model", False):
            if params["model_name"] != "test":
                if os.path.isdir(base):
                    print("Model already exists. Choose different model name.")
                    sys.exit(1)
                # else fall through and mkdir
            else:
                # test-mode: nuke any existing tree
                if os.path.isdir(base):
                    shutil.rmtree(base)
            # (re)create folders with 0o777
            os.makedirs(preds, mode=0o777, exist_ok=True)
        else:
            # just make sure it’s there
            os.makedirs(preds, mode=0o777, exist_ok=True)
    finally:
        # restore user’s umask
        os.umask(old_umask)

    # copy code/config in
    my_copy(base)

    # write params.txt (overwrite, not append)
    params_file = os.path.join(base, "params.txt")
    with open(params_file, "w") as f:
        for key, val in params.items():
            f.write(f"{key}: {val}\n")

    #––– force permissions on everything under base –––
    for root, dirs, files in os.walk(base):
        for d in dirs:
            os.chmod(os.path.join(root, d), 0o777)
        for fn in files:
            # give full read/write perms on files
            os.chmod(os.path.join(root, fn), 0o666)



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