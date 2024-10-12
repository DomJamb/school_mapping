import os
import time
import math
import random
import logging
import argparse
from tqdm import tqdm
import numpy as np

import pandas as pd
import torch
import geopandas as gpd
from owslib.wms import WebMapService
import sys
from pyproj import Transformer

sys.path.insert(0, "../utils/")
import data_utils
import config_utils
import cnn_utils

import threading
from queue import Queue
import time

from download_satellite_images import producer, consumer

import pickle
from PIL import Image
from torchvision import models, transforms
import torch.nn.functional as nnf

t1 = transforms.Compose(
    [
        transforms.CenterCrop(352),
    ]
)
t2 = transforms.Compose(
    [
        transforms.CenterCrop(500),
    ]
)

device = "cuda:1"
SEED = 42
logging.basicConfig(level=logging.INFO)

def get_images(c, category=1, n=10):
    """
    Downloads satellite images based on geographic data points.

    Args:
    - creds (dict): Credentials for accessing the satellite image service.
    - config (dict): Configuration settings.
    - category (str): Type of data category.
    - iso (str, optional): ISO code for the country. Defaults to None.
    - sample_size (int, optional): Number of samples to consider. Defaults to None.
    - src_crs (str, optional): Source coordinate reference system. Defaults to "EPSG:4326".
    - id_col (str, optional): Column name containing unique identifiers. Defaults to "UID".
    - name (str, optional): Name of the dataset. Defaults to "clean".
    - filename (str, optional): File name to load the data. Defaults to None.

    Returns:
    - None
    """

    dest_dir = '/mnt/ssd1/agorup/school_mapping/satellite_images/inference'
    out_file = os.path.join("inference_vietnam_difference.csv")


    df = pd.read_csv(out_file)
    df = df[df["category"]==category]
    #df = df.sample(n)
    for i in range(n):
        row = df.iloc[i]
        image = row["image"]
        separator_ind = image.rfind('-')
        image_file = f"{dest_dir}/{image[:separator_ind]}/{image[separator_ind+1:]}"
        img = Image.open(image_file).convert("RGB")
        img1 = t1(img)
        img2 = t2(img)

        img1.save(f"./images/{i}_352_{image}")
        img2.save(f"./images/{i}_500_{image}")




def main():
    # Parser
    parser = argparse.ArgumentParser(description="Satellite Image Download")
    parser.add_argument('-c', "--cnn_config", help="Config file", default="convnext_small")
    parser.add_argument("-e", "--exp", default="global_no_vietnam_500images_no_lowres_continuous_rotation_0-90_crop352_no_AMP")
    parser.add_argument("-f", "--fine_tune", default="vietnam_large")
    parser.add_argument('-d', "--device", help="device", default="cuda:1")
    args = parser.parse_args()

    # Load config
    cwd = os.path.dirname(os.getcwd())
    config_file = os.path.join(cwd, "configs", "cnn_configs", args.cnn_config + ".yaml")
    config = config_utils.load_config(config_file)
    device = args.device

    try:
        get_images(config, 1)
    except Exception as e:
        print(f"error: {e}")

    



if __name__ == "__main__":
    main()
