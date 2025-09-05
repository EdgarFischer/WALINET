params ={}

params["model_name"] =  "Final_No_B0_Correction"#"EXP_1"# "test"# 
params["path_to_model"] = "models/" + params["model_name"] + "/"
params["path_to_data"] = "data/"

params["train_subjects"]=['Vol3_Brisbane', 'Vol4_Brisbane', 'Vol5_Brisbane', 'Vol6_Brisbane', 'Vol2_London', 'Vol3_London', 'Vol4_London', 'Vol5_London', 'Vol6', 'Vol7', 'Vol8', 'Vol9']

params["val_subjects"]=['Vol5', 'Vol1_Brisbane', 'Vol1_London']

# Train Params
params["gpu"]=1
params["batch_size"] = 700#32
params["num_worker"] = 0
params["lr"] = 0.00025 # 0.0001#0.0001 
params["epochs"]=500
params["verbose"] = False#params["model_name"] == "test" #True #True #False #True #False #TrueFalse#
params["n_batches"] = -1
params["n_val_batches"] = -1
params["data_version"] = 'v_1.0'#'v3_2'

# LR Scheduler
params["milestones"] = [150, 300, 500]
params["gamma"] = 0.25

# Model Params
params["nLayers"] = 3#5#4
params["nFilters"] = 8#12#8#8#4#8#16#8
params["in_channels"] = 2 
params["out_channels"] = 2
params["dropout"] = 0.0#.005#.01# 0.005

params["clean_model"] = False # Only removes models called "test"
params["train"] = True
params["predict"] = False

params['preload'] = False
params['preload_model'] = ''
