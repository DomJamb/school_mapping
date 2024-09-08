import torch
import os
import pandas as pd
import geopandas as gpd
import argparse
import numpy as np

import sys
sys.path.insert(0, "../utils/")
import config_utils
import cnn_utils

import logging

import matplotlib.pyplot as plt

SEED = 42

if __name__ == "__main__":
    device = "cuda:1"

    parser = argparse.ArgumentParser(description="Data Cleaning Pipeline")
    parser.add_argument("-c", "--config", default="configs/cnn_configs/resnet18.yaml")
    parser.add_argument("-e", "--exp", default="all")
    parser.add_argument("-d", "--dataset", default="highres_old")
    args = parser.parse_args()

    exp = args.exp
    cwd = os.path.dirname(os.getcwd())
    config_file = os.path.join(cwd, args.config)
    c = config_utils.load_config(config_file)

    exp_name = f"{exp}_{c['config_name']}"
    exp_dir = os.path.join(cwd, c["exp_dir"], exp_name)
    model_file = os.path.join(exp_dir, f"{exp_name}.pth")
    dataset_name = args.dataset
    out_file = os.path.join(exp_dir, f"threshold_{dataset_name}.txt")

    if not os.path.exists(out_file):

        phases = ["train", "test"]
        data, data_loader, classes = cnn_utils.load_dataset(config=c, phases=phases, name=dataset_name)
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

        subdata = data["test"].dataset
        #print(f"{iso_code}: {len(subdata)}")

        classes_dict = {c["pos_class"] : 1, c["neg_class"]: 0}
        transforms = cnn_utils.get_transforms(size=c["img_size"])
        dataset =  cnn_utils.SchoolDataset(
                subdata,
                classes_dict,
                transforms["test"]
        )


        data_loader =  torch.utils.data.DataLoader(
                dataset,
                batch_size=c["batch_size"],
                num_workers=c["n_workers"],
                shuffle=False,
                drop_last=False
        )

    
        output = "threshold,F1,P,R\n"
        thresholds = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
        for t in thresholds:
            print(f"threshold: {t}")
            test_results, test_cm, test_preds = cnn_utils.evaluate(
                data_loader, classes, model, criterion, device, pos_label=1, logging=logging, threshold=t
            )

            output += f"{t},{round(test_results['f1_score'], 2)},{round(test_results['precision_score'], 2)},{round(test_results['recall_score'], 2)}\n"

        f = open(out_file, "w")
        f.write(output)
        f.close()



