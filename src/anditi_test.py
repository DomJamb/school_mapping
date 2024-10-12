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

imagenet_mean, imagenet_std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
t = transforms.Compose(
    [
        
        #transforms.Resize(size),
        #transforms.CenterCrop(size),
        transforms.CenterCrop(500),
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean, imagenet_std),
    ]
)


device = "cuda:0"
SEED = 42
logging.basicConfig(level=logging.INFO)


def inference(c, exp, finetune):
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
    
    f = "/mnt/ssd1/agorup/school_mapping/inference_data/Anditi_filtered_schools_2-3857.csv"
    data = pd.read_csv(f)
    dest_dir = '/mnt/ssd1/agorup/school_mapping/satellite_images/anditi/large'

    images_school = []
    for i in range(len(data)):
        image_file = f"{dest_dir}/{i}.jpeg"
        images_school.append(image_file)

    cwd = os.path.dirname(os.getcwd())
    exp_name = f"{exp}"
    exp_dir = os.path.join(cwd, c["exp_dir"], exp_name)
    if finetune != "":
        exp_dir = os.path.join(exp_dir, f"fine_tune_{finetune}")
    print(exp_dir)
    model_file = os.path.join(exp_dir, f"model.pth")
    if not os.path.exists(model_file):
        model_file = os.path.join(exp_dir, f"{exp_name}.pth")
    out_file_school = os.path.join(exp_dir, "anditi_school.csv")
    out_file_non_school = os.path.join(exp_dir, "anditi_non_school.csv")

    if not os.path.exists(exp_dir) or not os.path.exists(model_file):
        print("wrong paths")
        return

    #if not os.path.exists(out_file_school):

    classes = ['school', 'non_school']
    model, criterion, optimizer, scheduler = cnn_utils.load_model(
        n_classes=len(classes),
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

    

    model.load_state_dict(torch.load(model_file, map_location=device))
    model = model.to(device)
    model.eval()

    preds_school = []
    df_school = pd.DataFrame(columns=["image", "pred", "lon", "lat"])
    for i in tqdm(range(len(images_school))):
        image_file = images_school[i]
        img = Image.open(image_file).convert("RGB")
        img = t(img)
        img = img.unsqueeze(0).to(device)
        out = model(img)
        soft_outputs = nnf.softmax(out, dim=1)
        positive_prob = soft_outputs[:, 1]
        pred = int(positive_prob > 0.5)
        preds_school.append(pred)

        lon = data.iloc[i]["longitude"]
        lat = data.iloc[i]["latitude"]
        row = {"image": f"{i}.jpeg","pred": positive_prob[0].data.cpu().numpy(), "lon":lon, "lat":lat}
        df_school.loc[i] = row

    phases = ["train", "test"]
    data, data_loader, classes = cnn_utils.load_dataset(config=c, phases=phases, name = "vietnam")
    data = data["test"].dataset
    data = data[data['class']=="non_school"]
    data = data[data['clean']==0]
    data = data.sample(n=len(images_school))

    images_non_school = []
    for i, row in data.iterrows():
        image_file = f"/mnt/ssd1/agorup/school_mapping/satellite_images/VNM/non_school/{row['UID']}.jpeg"
        images_non_school.append(image_file)

    preds_non_school = []
    df_non_school = pd.DataFrame(columns=["image", "pred", "lon", "lat"])
    for i in tqdm(range(len(images_non_school))):
        image_file = images_non_school[i]
        img = Image.open(image_file).convert("RGB")
        img = t(img)
        img = img.unsqueeze(0).to(device)
        out = model(img)
        soft_outputs = nnf.softmax(out, dim=1)
        positive_prob = soft_outputs[:, 0]
        pred = int(positive_prob > 0.5)
        preds_non_school.append(pred)

        lon = data.iloc[i]["geometry"].x
        lat = data.iloc[i]["geometry"].y
        row = {"image": f"{data.iloc[i]['UID']}.jpeg","pred": positive_prob[0].data.cpu().numpy(), "lon":lon, "lat":lat}
        df_non_school.loc[i] = row

    accuracy_school = np.sum(preds_school) / len(images_school)
    print(accuracy_school)

    log_string = ""
    accuracy = (np.sum(preds_school) + np.sum(preds_non_school)) / (len(images_school) + len(images_non_school))
    log_string += f"Accuracy: {accuracy}\n"
    precision = np.sum(preds_school) / (np.sum(preds_school) + (len(images_non_school) - np.sum(preds_non_school)))
    log_string += f"Precision: {precision}\n"
    recall = np.sum(preds_school) / len(images_school)
    log_string += f"Recall: {recall}\n"
    f1 = (2 * precision * recall) / (precision + recall)
    log_string += f"F1: {f1}\n"
    log_string += "\n\n"
    log_string += "Confusion matrix:\n"
    log_string += f"{np.sum(preds_school)},{len(images_school) - np.sum(preds_school)}\n"
    log_string += f"{len(images_non_school) - np.sum(preds_non_school)},{np.sum(preds_non_school)}\n"

    log_file = os.path.join(exp_dir, "anditi_results.txt")
    f = open(log_file, "w")
    f.write(log_string)
    f.close()

    df_school = df_school.sort_values("pred", ascending=False)
    df_school.to_csv(out_file_school, index=False)

    df_non_school = df_non_school.sort_values("pred", ascending=False)
    df_non_school.to_csv(out_file_non_school, index=False)





def main():
    # Parser
    parser = argparse.ArgumentParser(description="Satellite Image Download")
    parser.add_argument('-c', "--cnn_config", help="Config file", default="convnext_small")
    parser.add_argument("-e", "--exp", default="global_no_vietnam_500images_no_lowres_continuous_rotation_0-90_crop352_no_AMP_convnext_small")
    parser.add_argument("-f", "--fine_tune", default="vietnam_uninhabited")
    parser.add_argument('-d', "--device", help="device", default="cuda:0")
    args = parser.parse_args()

    # Load config
    cwd = os.path.dirname(os.getcwd())
    config_file = os.path.join(cwd, "configs", "cnn_configs", args.cnn_config + ".yaml")
    config = config_utils.load_config(config_file)
    device = args.device

    try:
        inference(config, args.exp, args.fine_tune)
    except Exception as e:
        print(f"error: {e}")

    



if __name__ == "__main__":
    main()
