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
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

imagenet_mean, imagenet_std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
t1 = transforms.Compose(
    [
        
        #transforms.Resize(size),
        #transforms.CenterCrop(size),
        transforms.CenterCrop(500),
    ]
)
t2 = transforms.Compose(
    [
        
        #transforms.Resize(size),
        #transforms.CenterCrop(size),
        transforms.CenterCrop(500),
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean, imagenet_std),
    ]
)
t3 = transforms.Compose(
    [
        
        #transforms.Resize(size),
        #transforms.CenterCrop(size),
        transforms.CenterCrop(500),
        transforms.ToTensor(),
        
    ]
)

device = "cuda:1"
SEED = 42
logging.basicConfig(level=logging.INFO)

def get_images(c, exp, finetune="", n=10):
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

    cwd = os.path.dirname(os.getcwd())
    exp_name = f"{exp}_{c['config_name']}"
    exp_dir = os.path.join(cwd, c["exp_dir"], exp_name)
    if finetune != "":
        exp_dir = os.path.join(exp_dir, f"fine_tune_{finetune}")
    print(exp_dir)
    out_file = os.path.join(exp_dir, "inference_vietnam.csv")

    model, criterion, optimizer, scheduler = cnn_utils.load_model(
        n_classes=2,
        model_type=c["model"],
        pretrained=c["pretrained"],
        scheduler_type=c["scheduler"],
        optimizer_type=c["optimizer"],
        label_smoothing=c["label_smoothing"],
        lr=c["lr"],
        momentum=c["momentum"],
        gamma=c["gamma"],
        step_size=c["step_size"],
        patience=c["patience"],
        dropout=c["dropout"],
        device=device,
    )

    model_file = os.path.join(exp_dir, exp_name + ".pth")
    if not os.path.exists(model_file):
        model_file = os.path.join(exp_dir, "model.pth")
    model.load_state_dict(torch.load(model_file, map_location=device))
    model = model.to(device)

    if not os.path.exists(exp_dir):
        return

    cot = 1
    targets = [ClassifierOutputTarget(cot)]
    target_layers = [model.features[-1]]
    cam = GradCAM(model=model, target_layers=target_layers)
    df = pd.read_csv(out_file)
    df = df[df['pred'] >= 0.5]
    df = df.sample(n)
    for i in range(n):
        row = df.iloc[i]
        image = row["image"]
        separator_ind = image.rfind('-')
        image_file = f"{dest_dir}/{image[:separator_ind]}/{image[separator_ind+1:]}"
        img = Image.open(image_file).convert("RGB")
        img1 = t1(img)
        img1.save(f"./images/{i}_{image}")
        img1 = t3(img)

        img2 = t2(img)
        img2 = img2.unsqueeze(0).to(device)
        img2 = img2.to(device)
        grayscale_cam = cam(input_tensor=img2, targets=targets)
        grayscale_cam = grayscale_cam[0, :]
        rgb_img = img1.cpu().numpy().transpose(1, 2, 0)
        visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
        img = Image.fromarray(visualization)
        img.save(f"./images/{i}_gradcam_{image}")




        





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
        get_images(config, args.exp, args.fine_tune)
    except Exception as e:
        print(f"error: {e}")

    



if __name__ == "__main__":
    main()
