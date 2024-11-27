# Copyright (c) 2024 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0


"""File containing the paths will be used in the code."""

import os
import sys
from easydict import EasyDict

CONF = EasyDict()

# Get the current working directory or use environment variables for paths
CONF.PATH = EasyDict()
CONF.PATH.HOME = os.getenv("HOME_PATH", os.path.join("/mnt/scratch/3DSG"))  # Home directory
CONF.PATH.BASE = os.getenv("BASE_PATH", os.path.join(CONF.PATH.HOME, "Open3DSG-main"))  # Base directory
CONF.PATH.DATA = os.getenv("DATA_PATH", os.path.join(CONF.PATH.HOME, "Datasets"))  # Data directory

# Append to sys.path for module imports
for _, path in CONF.PATH.items():
    sys.path.append(path)

# Original Datasets
CONF.PATH.R3SCAN_RAW = os.path.join(CONF.PATH.DATA, "3RScan")  # 3RScan original dataset directory
CONF.PATH.SCANNET_RAW = os.path.join("/mnt/datasets/scannet")  # ScanNet original dataset directory
CONF.PATH.SCANNET_RAW3D = os.path.join("/mnt/scratch/3DSG/Datasets/SCANNET")  # ScanNet 3D dataset directory
CONF.PATH.SCANNET_RAW2D = os.path.join(CONF.PATH.DATA, "SCANNET","scannet_2d")  # ScanNet 2D dataset directory

# Processed Dataset Paths
CONF.PATH.DATA_OUT = os.getenv("DATA_OUT_PATH", os.path.join(CONF.PATH.DATA, "output"))  # Output directory for processed datasets
CONF.PATH.R3SCAN = os.path.join(CONF.PATH.DATA_OUT, "datasets", "OpenSG_3RScan")  # Processed 3RScan dataset
CONF.PATH.SCANNET = os.path.join(CONF.PATH.DATA_OUT, "datasets", "OpenSG_ScanNet")  # Processed ScanNet dataset
CONF.PATH.CHECKPOINTS = os.path.join(CONF.PATH.DATA_OUT, "checkpoints")  # Checkpoints directory
CONF.PATH.FEATURES = os.path.join(CONF.PATH.DATA_OUT, "features")  # Features directory

# MLOps Directories
CONF.PATH.MLOPS = os.path.join(CONF.PATH.BASE, "mlops")  # MLOps directory
CONF.PATH.MLFLOW = os.path.join(CONF.PATH.MLOPS, "opensg", "mlflow")  # MLFlow data directory
CONF.PATH.TENSORBOARD = os.path.join(CONF.PATH.MLOPS, "opensg", "tensorboards")  # Tensorboard data directory

# Ensure all paths exist; create missing directories
for key, path in CONF.PATH.items():
    if not os.path.exists(path):
        print(f"Creating missing directory: {path}")
        os.makedirs(path, exist_ok=True)
    assert os.path.exists(path), f"{path} does not exist"
