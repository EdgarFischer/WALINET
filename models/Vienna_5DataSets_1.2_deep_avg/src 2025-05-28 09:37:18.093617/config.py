params ={}

params["model_name"] =  "Vienna_5DataSets_1.2_deep_avg"#"EXP_1"# "test"# 
params["path_to_model"] = "models/" + params["model_name"] + "/"
params["path_to_data"] = "data/"

params["train_subjects"]=['Vol6','Vol7','Vol8','Vol9'] #

params["val_subjects"]=['Vol5']

# Train Params
params["gpu"]=1
params["batch_size"] = 1000#32
params["num_worker"] = 15
params["lr"] = 0.00025 # 0.0001#0.0001 
params["epochs"]=1000
params["verbose"] = False#params["model_name"] == "test" #True #True #False #True #False #TrueFalse#
params["n_batches"] = -1
params["n_val_batches"] = -1
params["data_version"] = 'v_1.0_averaged'#'v3_2'

# LR Scheduler
params["milestones"] = [400, 700, 1000]
params["gamma"] = 0.25

# Model Params
params["nLayers"] = 7#5#4
params["nFilters"] = 8#12#8#8#4#8#16#8
params["in_channels"] = 2 
params["out_channels"] = 2
params["dropout"] = 0.0#.005#.01# 0.005

params["clean_model"] = True # Only removes models called "test"
params["train"] = True
params["predict"] = False

params['preload'] = False
params['preload_model'] = ''
