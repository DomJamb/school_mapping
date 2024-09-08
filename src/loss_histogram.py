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
    device = "cuda:0"

    parser = argparse.ArgumentParser(description="Data Cleaning Pipeline")
    parser.add_argument("-c", "--config", default="configs/cnn_configs/resnet18.yaml")
    parser.add_argument("-e", "--exp", default="all_no_resize")
    args = parser.parse_args()

    exp = args.exp
    cwd = os.path.dirname(os.getcwd())
    config_file = os.path.join(cwd, args.config)
    c = config_utils.load_config(config_file)

    exp_name = f"{exp}_{c['config_name']}"
    exp_dir = os.path.join(cwd, c["exp_dir"], exp_name)
    model_file = os.path.join(exp_dir, f"{exp_name}.pth")
    out_file = os.path.join(exp_dir, "train_losses.npy")

    if not os.path.exists(out_file):

        phases = ["train", "test"]
        data, data_loader, classes = cnn_utils.load_dataset(config=c, phases=phases, name="highres_old")
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

        subdata = data["train"].dataset
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


        test_results, test_cm, test_preds = cnn_utils.evaluate(
            data_loader, classes, model, criterion, device, pos_label=1, logging=logging
        )

        losses = []

        for ind in test_preds.index:
            if test_preds["y_true"][ind] == test_preds["y_preds"][ind]:
                losses.append(1 - test_preds["y_probs"][ind])
            else:
                losses.append(test_preds["y_probs"][ind])

        losses = np.array(losses)
        file = open(out_file, "wb")
        np.save(file, losses)

        mask = losses < 0.75
        data_filtered = subdata[mask]
        subdata2 = data["test"].dataset
        data_filtered = pd.concat([data_filtered, subdata2])
        filtered_file = os.path.join(cwd, "data", "vectors", "train", "highres_old_filtered_train.geojson")
        data_filtered.to_file(filtered_file,  driver="GeoJSON")

    file = open(out_file, 'rb')
    losses = np.load(file)


    #plt.hist(losses, bins=200)
    #plt.savefig(os.path.join(exp_dir, "loss_hist.png"))



