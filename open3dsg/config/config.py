# Copyright (c) 2024 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

"""File containing the paths will be used in the code."""

import os
import sys
from easydict import EasyDict

# IMP: CHANGE NAME TO ADAPT PATHS
name = "Elena"
use_subset = True
CONF = EasyDict()
CONF.PATH = EasyDict()

# Invidual paths depending where you mounted the drives
#CONF.PATH.DATASETS --> mounted datasets folder from cluster
#CONF.PATH.DATA_OUT --> mounted output folder from projects folder
#CONF.PATH.DATA --> mounted data folder from projects folder
if name == "Noah":
    CONF.PATH.HOME = "/workspace"                           # Your home directory
    CONF.PATH.BASE = "/workspace"                           # OpenSG directory
    CONF.PATH.DATA = "../open3dsg/data"                     # Root path for datasets
    CONF.PATH.DATASETS = "../datasets"                      # Root path for datasets
    CONF.PATH.DATA_OUT = "../open3dsg/output"
if name =='Elena':
    CONF.PATH.HOME = "/mnt/scratch"                                                        
    CONF.PATH.BASE = "/mnt/scratch/Practical-Open3DSG"                                               
    CONF.PATH.DATA = "/mnt/projects/open3dsg/data"  
    CONF.PATH.DATASETS = "/mnt/datasets"                                         
    CONF.PATH.DATA_OUT = "/mnt/projects/open3dsg/output"                                    
elif name =='Ayaka':
    CONF.PATH.HOME = "/mnt/scratch"                                                       
    CONF.PATH.BASE = "/mnt/scratch/p3dcv/Practical-Open3DSG"                               
    CONF.PATH.DATA = "/mnt/projects/open3dsg/data" 
    CONF.PATH.DATASETS = "/mnt/datasets"                                        
    CONF.PATH.DATA_OUT = "/mnt/projects/open3dsg/output"        
elif name =='Sebastian':
    CONF.PATH.HOME = "/mnt/scratch"                                                       
    CONF.PATH.BASE = "/mnt/scratch/p3dcv/Practical-Open3DSG"                               
    CONF.PATH.DATA = "/mnt/projects/open3dsg/data"                                         
    CONF.PATH.DATASETS = "/mnt/datasets"
    CONF.PATH.DATA_OUT = "/mnt/projects/open3dsg/output" 

# Append to syspath
for _, path in CONF.PATH.items():
    sys.path.append(path)

# ----------------- Original Datasets -----------------
CONF.PATH.R3SCAN_RAW = os.path.join(CONF.PATH.DATA, "3RScan")                               # 3RScan original dataset directory
CONF.PATH.SCANNET_RAW_DATASETS = os.path.join(CONF.PATH.DATASETS, "scannet", "scans")       # ScanNet original dataset directory
CONF.PATH.SCANNET_RAW_PROJECTS = os.path.join(CONF.PATH.DATA, "SCANNET")                    # ScanNet original dataset directory
if use_subset: 
    print('IMPORTANT: Using subset ...')
    CONF.PATH.SCANNET_RAW_PROJECTS = os.path.join(CONF.PATH.DATA, "subset_scannet")                # ScanNet subset original dataset directory


# ----------------- Processed Dataset -----------------
# CONF.PATH.R3SCAN = os.path.join(CONF.PATH.DATA, "OpenSG_3RScan")
# CONF.PATH.SCANNET = os.path.join(CONF.PATH.DATA, "OpenSG_ScanNet")
# Output directory for processed datasets
CONF.PATH.R3SCAN = os.path.join(CONF.PATH.DATA_OUT, "datasets", "OpenSG_3RScan")            # Output directory for processed 3RScan dataset
CONF.PATH.SCANNET = os.path.join(CONF.PATH.DATA_OUT, "datasets", "OpenSG_ScanNet")          # Output directory for processed ScanNet dataset
if use_subset:
    print('IMPORTANT: Using subset ...')
    CONF.PATH.SCANNET = os.path.join(CONF.PATH.DATA_OUT, "datasets", "OpenSG_ScanNets")          # Output directory for processed ScanNet dataset
CONF.PATH.CHECKPOINTS = os.path.join(CONF.PATH.DATA_OUT, "checkpoints")
CONF.PATH.FEATURES = os.path.join(CONF.PATH.DATA_OUT, "features")


# ----------------- MLOps -----------------
CONF.PATH.MLOPS = os.path.join(CONF.PATH.BASE, "mlops")                                     # MLOps directory
CONF.PATH.MLFLOW = os.path.join(CONF.PATH.MLOPS, "opensg", "mlflow")                        # Output directory for MLFlow data
CONF.PATH.TENSORBOARD = os.path.join(CONF.PATH.MLOPS, "opensg", "tensorboards")             # Output directory for Tensorboard data

for _, path in CONF.PATH.items():
    assert os.path.exists(path), f"{path} does not exist"
