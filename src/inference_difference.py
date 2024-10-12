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

t = transforms.Compose(
    [
        
        #transforms.Resize(size),
        #transforms.CenterCrop(size),
        transforms.CenterCrop(500),
        #transforms.ToTensor(),
    ]
)
pyproj_transformer = Transformer.from_crs("EPSG:3857", "EPSG:4326")

device = "cuda:1"
SEED = 42
logging.basicConfig(level=logging.INFO)

def get_images(c):
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
    f = open("/mnt/ssd1/agorup/school_mapping/inference_data/district_name_to_bboxes_3857-500px.pkl", 'rb')
    data = pickle.load(f)

    dest_dir = '/mnt/ssd1/agorup/school_mapping/satellite_images/inference'

    images = []
    for district in data:
        for index in range(len(data[district])):
            image_file = f"{dest_dir}/{district}/{index}.jpeg"
            images.append(image_file)

    dest_dir = '/mnt/ssd1/agorup/school_mapping/satellite_images/inference'

    cwd = os.path.dirname(os.getcwd())
    exp_dir1 = os.path.join(cwd, c["exp_dir"], "global_no_vietnam_500images_no_lowres_continuous_rotation_0-90_crop352_no_AMP_convnext_small", "fine_tune_vietnam")
    out_file1 = os.path.join(exp_dir1, "inference_vietnam.csv")
    exp_dir2 = os.path.join(cwd, c["exp_dir"], "global_no_vietnam_500images_no_lowres_continuous_rotation_0-90_crop352_no_AMP_convnext_small", "fine_tune_vietnam_large")
    out_file2 = os.path.join(exp_dir2, "inference_vietnam.csv")
    

    df1 = pd.read_csv(out_file1)
    df2 = pd.read_csv(out_file2)
    # 0 -> 352 Y, 500 Y
    # 1 -> 352 Y, 500 N
    # 2 -> 352 N, 500 Y
    # 3 -> 352 N, 500 N
    df_new = pd.DataFrame(columns=["image", "category", "pred1", "pred2", "diff", "lon", "lat"])
    
    for i in tqdm(range(len(images))):
        image_file = images[i]
        p, index = os.path.split(image_file)
        p, district = os.path.split(p)
        image = f"{district}-{index}"
        pred1 = df1[df1["image"] == image].iloc[0]["pred"]
        pred2 = df2[df2["image"] == image].iloc[0]["pred"]
        diff = abs(pred1 - pred2)
        category = (2 * (pred1 < 0.5) + 1 * (pred2 < 0.5)).astype(int)
        lon = df1[df1["image"] == image].iloc[0]["lon"]
        lat = df1[df1["image"] == image].iloc[0]["lat"]
        row = {"image":image, "category":category, "pred1":pred1, "pred2":pred2, "diff":diff, "lon":lon, "lat":lat}
        df_new.loc[i] = row

    df_new = df_new.sort_values(by=["category", "diff"], ascending=[True,False])
    df_new.to_csv("inference_vietnam_difference.csv", index=False)



def main():
    # Parser
    parser = argparse.ArgumentParser(description="Satellite Image Download")
    parser.add_argument('-c', "--cnn_config", help="Config file", default="convnext_small")
    args = parser.parse_args()

    # Load config
    cwd = os.path.dirname(os.getcwd())
    config_file = os.path.join(cwd, "configs", "cnn_configs", args.cnn_config + ".yaml")
    config = config_utils.load_config(config_file)

    try:
        get_images(config)
    except Exception as e:
        print(f"error: {e}")

    



if __name__ == "__main__":
    main()
