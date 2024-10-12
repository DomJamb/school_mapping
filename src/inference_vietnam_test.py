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
from torch.utils.data import Dataset

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
pyproj_transformer = Transformer.from_crs("EPSG:3857", "EPSG:4326")

class SchoolDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, index):
        item = self.dataset.iloc[index]
        filepath= item["filepath"]
        image = Image.open(filepath).convert("RGB")

        if self.transform:
            x = self.transform(image)

        p, image_index = os.path.split(filepath)
        p, district = os.path.split(p)
        image_index = image_index[:-5]
        lon = item["lon"]
        lat = item["lat"]
        image_name = f"{district}-{image_index}.jpeg"
        image.close()
        return x, image_name, lon, lat

    def __len__(self):
        return len(self.dataset)

def inference(c, exp, finetune): 
    # f = open("/mnt/ssd1/agorup/school_mapping/inference_data/district_name_to_bboxes_3857-500px.pkl", 'rb')
    # dest_dir = '/mnt/ssd1/agorup/school_mapping/satellite_images/inference/large'
    f = "/mnt/ssd1/agorup/school_mapping/inference_data/inference_filtered_ghsl.csv"
    dest_dir = '/mnt/ssd1/agorup/school_mapping/satellite_images/inference/large'
    images = []

    # data = pickle.load(f)
    # for district in data:
    #     for index in range(len(data[district])):
    #         row = data[district][index]
    #         image_file = f"{dest_dir}/{district}/{index}.jpeg"
    #         #df_images.loc[len(df_images)] = {"filepath":image_file}
    #         lon = (row["min_lon"] + row["max_lon"]) / 2
    #         lat = (row["min_lat"] + row["max_lat"]) / 2
    #         geom = pyproj_transformer.transform(lon,lat)
    #         geom_lon = f"{geom[0]}N" if geom[0] >= 0 else f"{-1*geom[0]}S"
    #         geom_lat = f"{geom[1]}E" if geom[1] >= 0 else f"{-1*geom[1]}W"
    #         images.append({"filepath":image_file, "lon":geom_lon, "lat":geom_lat})

    df = pd.read_csv(f)
    data = df
    for i, row in df.iterrows():
        district = row["district"]
        image = f"{row['index']}.jpeg"
        image_file = f"{dest_dir}/{district}/{image}"
        lon = row["longitude"]
        lat = row["latitude"]
        geom = pyproj_transformer.transform(lon,lat)
        geom_lon = f"{geom[0]}N" if geom[0] >= 0 else f"{-1*geom[0]}S"
        geom_lat = f"{geom[1]}E" if geom[1] >= 0 else f"{-1*geom[1]}W"
        images.append({"filepath":image_file, "lon":geom_lon, "lat":geom_lat})

    df_images = pd.DataFrame(images)
    dataset = SchoolDataset(df_images, t)
    data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=c["batch_size"],
            num_workers=c["n_workers"],
            shuffle=False,
            drop_last=False
        )

    cwd = os.path.dirname(os.getcwd())
    exp_name = f"{exp}"
    exp_dir = os.path.join(cwd, c["exp_dir"], exp_name)
    if finetune != "":
        exp_dir = os.path.join(exp_dir, f"fine_tune_{finetune}")
    print(exp_dir)
    model_file = os.path.join(exp_dir, f"model.pth")
    if not os.path.exists(model_file):
        model_file = os.path.join(exp_dir, f"{exp_name}.pth")
    out_file = os.path.join(exp_dir, "inference_vietnam_filtered_ghsl.csv")
    #out_file = os.path.join(exp_dir, "inference_vietnam_overlap.csv")

    if not os.path.exists(exp_dir) or not os.path.exists(model_file):
        print("wrong paths")
        return

    if not os.path.exists(out_file) or True:

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

        df = pd.DataFrame(columns=["image", "pred", "lon", "lat"])
        # for i in tqdm(range(len(images))):
        #     image_file = images[i]
        #     img = Image.open(image_file).convert("RGB")
        #     img = t(img)
        #     img = img.unsqueeze(0).to(device)
        #     out = model(img)
        #     soft_outputs = nnf.softmax(out, dim=1)
        #     positive_prob = soft_outputs[:, 1]
        #     p, index = os.path.split(image_file)
        #     p, district = os.path.split(p)

        #     lon = (data[district][int(index[:-5])]["min_lon"] + data[district][int(index[:-5])]["max_lon"]) / 2
        #     lat = (data[district][int(index[:-5])]["min_lat"] + data[district][int(index[:-5])]["max_lat"]) / 2
        #     geom = pyproj_transformer.transform(lon,lat)
        #     geom1 = f"{geom[0]}N" if geom[0] >= 0 else f"{-1*geom[0]}S"
        #     geom2 = f"{geom[1]}E" if geom[1] >= 0 else f"{-1*geom[1]}W"

        #     row = {"image": f"{district}-{index}","pred": positive_prob[0].data.cpu().numpy(), "lon":geom1, "lat":geom2}
        #     df.loc[i] = row

        for inputs, image_names, lons, lats in tqdm(data_loader, total=len(data_loader)):
            inputs = inputs.to(device)

            with torch.set_grad_enabled(False):
                #with torch.autocast(device_type="cuda", dtype=torch.float16):
                outputs = model(inputs)
                soft_outputs = nnf.softmax(outputs, dim=1)
                positive_probs = soft_outputs[:, 1]
            
            batch_dict = {
                "image": image_names,
                "pred": positive_probs.cpu().view(-1).numpy(),
                "lon": lons,
                "lat": lats
            }
            batch_df = pd.DataFrame(batch_dict)
            df = pd.concat([df, batch_df], ignore_index=True)



        df = df.sort_values("pred", ascending=False)
        df.to_csv(out_file, index=False)




def main():
    # Parser
    parser = argparse.ArgumentParser(description="Satellite Image Download")
    parser.add_argument('-c', "--cnn_config", help="Config file", default="convnext_small")
    parser.add_argument("-e", "--exp", default="global_no_vietnam_500images_no_lowres_continuous_rotation_0-90_crop352_no_AMP_convnext_small/fine_tune_vietnam_large")
    parser.add_argument("-f", "--fine_tune", default="anditi1")
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
