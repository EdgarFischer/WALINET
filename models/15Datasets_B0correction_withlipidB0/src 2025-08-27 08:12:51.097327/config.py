params ={}

params["model_name"] =  "15Datasets_B0correction_withlipidB0"#"EXP_1"# "test"# 
params["path_to_model"] = "models/" + params["model_name"] + "/"
params["path_to_data"] = "data/"

params["train_subjects"]=['Vol3_Brisbane_B0_corrected', 'Vol4_Brisbane_B0_corrected','Vol5_Brisbane_B0_corrected','Vol6_Brisbane_B0_corrected',
            'Vol2_London_B0_corrected','Vol3_London_B0_corrected','Vol4_London_B0_corrected','Vol5_London_B0_corrected',
            'Vol6_B0_corrected','Vol7_B0_corrected','Vol8_B0_corrected','Vol9_B0_corrected'] #

params["val_subjects"]=['Vol1_Brisbane_B0_corrected','Vol1_London_B0_corrected','Vol5_B0_corrected']

# Train Params
params["gpu"]=1
params["batch_size"] = 700#32
params["num_worker"] = 15
params["lr"] = 0.00025 # 0.0001#0.0001 
params["epochs"]=500
params["verbose"] = False#params["model_name"] == "test" #True #True #False #True #False #TrueFalse#
params["n_batches"] = -1
params["n_val_batches"] = -1
params["data_version"] = 'v_2.0_noNans'#'v3_2'

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
