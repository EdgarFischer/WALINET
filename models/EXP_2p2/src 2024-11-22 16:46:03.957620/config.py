params ={}

params["model_name"] =  "test"#"EXP_1"# "test"# 
params["path_to_model"] = "/workspace/walinet/models/" + params["model_name"] + "/"
params["path_to_data"] = "/workspace/walinet/PaulTrainData/"

params["train_subjects"]=['3DMRSIMAP_Vol_04_A_1_2024-09-06_L2_0p0005',
                          '3DMRSIMAP_Vol_05_A_1_2024-09-02_L2_0p0005',
                          '3DMRSIMAP_Vol_06_A_1_2024-08-22_L2_0p0005',
                          '3DMRSIMAP_Vol_08_A_1_2024-09-07_L2_0p0005',
                          '3DMRSIMAP_Vol_09_A_1_2024-09-03_L2_0p0005'] #
#params["train_subjects"]=['3DMRSIMAP_Vol_09_A_1_2024-09-03_L2_0p0005']

params["val_subjects"]=['3DMRSIMAP_Vol_06_A_1_2024-08-22_L2_0p0005']

# Train Params
params["gpu"]=2
params["batch_size"] = 256#32
params["num_worker"] = 15
params["lr"] = 0.0001 # 0.0001#0.0001 
params["epochs"]=1000
params["verbose"] = params["model_name"] == "test" #True #True #False #True #False #TrueFalse#
params["n_batches"] = -1
params["n_val_batches"] = -1
params["data_version"] = 'v3_2'

# LR Scheduler
params["milestones"] = [400, 600, 800]
params["gamma"] = 0.25

# Model Params
params["nLayers"] = 4#5#4
params["nFilters"] = 8#12#8#8#4#8#16#8
params["in_channels"] = 2 
params["out_channels"] = 2
params["dropout"] = 0.0#.005#.01# 0.005

params["clean_model"] = True # Only removes models called "test"
params["train"] = True
params["predict"] = False

params['preload'] = False
params['preload_model'] = ''