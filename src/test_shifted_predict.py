import os
import time
import argparse
from collections import Counter
import torch

import sys
sys.path.insert(0, "../utils/")
import config_utils
import cnn_utils
import eval_utils
import wandb
import logging

import torch.nn as nn
import numpy as np

# Get device
cwd = os.path.dirname(os.getcwd())
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
logging.info(f"Device: {device}")

SEED = 42

def test(config, exp_name="", dataset_name=None):

    if exp_name is None or exp_name == "":
        exp_name = f"{dataset_name}_{c['config_name']}" if dataset_name is not None else f"{c['config_name']}"
    else:
        exp_name = f"{dataset_name}_{exp_name}_{c['config_name']}" if dataset_name is not None else f"{exp_name}_{c['config_name']}"
    exp_dir = os.path.join(cwd, c["exp_dir"], exp_name)

    phases = ["train", "test"]
    data, data_loader, classes = cnn_utils.load_dataset(config=c, phases=phases, name = dataset_name)

    

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

    model_file = os.path.join(exp_dir, f"{exp_name}.pth")
    model.load_state_dict(torch.load(model_file, map_location=device))
    model = model.to(device)

    # test_results, test_cm, test_preds = cnn_utils.evaluate(
    #     data_loader["test"], classes, model, criterion, device, pos_label=1, wandb=wandb, logging=logging
    # )
    # test_preds.to_csv(os.path.join(exp_dir, f"{exp_name}.csv"), index=False)

    # # Save results in experiment directory
    # eval_utils._save_files(test_results, test_cm, exp_dir)

    subresults_dir = os.path.join(exp_dir, "shifted_test")

    #print(data["test"].dataset.columns)
    subdata = data["test"].dataset
    #print(f"{iso_code}: {len(subdata)}")

    classes_dict = {config["pos_class"] : 1, config["neg_class"]: 0}
    transforms = cnn_utils.get_transforms(size=config["img_size"])
    dataset =  cnn_utils.SchoolDataset(
            subdata
            .sample(frac=1, random_state=SEED)
            .reset_index(drop=True),
            classes_dict,
            transforms["test"]
    )


    data_loader =  torch.utils.data.DataLoader(
            dataset,
            batch_size=config["batch_size"],
            num_workers=config["n_workers"],
            shuffle=False,
            drop_last=False
    )

    if len(data_loader) == 0:
        return

    if not os.path.exists(subresults_dir):
        os.makedirs(subresults_dir)
    test_results, test_cm, test_preds = cnn_utils.evaluate(
        data_loader, classes, model, criterion, device, pos_label=1, wandb=wandb, logging=logging
    )
    test_preds.to_csv(os.path.join(subresults_dir, f"preds.csv"), index=False)

    # Save results in experiment directory
    eval_utils._save_files(test_results, test_cm, subresults_dir)

if __name__ == "__main__":
    # Parser
    parser = argparse.ArgumentParser(description="Model Training")
    parser.add_argument("--cnn_config", help="Config file", default="resnet18")
    parser.add_argument("--iso", help="ISO code", default=[
        'ATG', 'AIA', 'YEM', 'SEN', 'BWA', 'MDG', 'BEN', 'BIH', 'BLZ', 'BRB', 
        'CRI', 'DMA', 'GHA', 'GIN', 'GRD', 'HND', 'HUN', 'KAZ', 'KEN', 'KIR', 
        'KNA', 'LCA', 'MNG', 'MSR', 'MWI', 'NAM', 'NER', 'NGA', 'PAN', 'RWA', 
        'SLE', 'SLV', 'SSD', 'THA', 'TTO', 'UKR', 'UZB', 'VCT', 'VGB', 'ZAF', 
        'ZWE', 'BRA'
    ], nargs='+')
    parser.add_argument("--device", help="device", default="cuda:1")
    parser.add_argument('-e', "--exp_name", default="")
    parser.add_argument('--dataset', default="all")
    args = parser.parse_args()

    device = torch.device(args.device)
    logging.info(f"config: {args.cnn_config}")
    logging.info(f"Args Device: {device}")
    logging.info(f"test: {args.test}")
    logging.info(f"exp_name: {args.dataset}_{args.exp_name}")
    logging.info(f"dataset: {args.dataset}")

    # Load config
    config_file = os.path.join(cwd, "configs", "cnn_configs", args.cnn_config + ".yaml")
    c = config_utils.load_config(config_file)
    iso_codes = args.iso
    # iso_codes = [
    #     "THA", 'KHM', 'LAO', 'IDN', 'PHL', 'MYS', 'MMR', 'BGD', 'BRN'
    # ]
    iso_codes = [
        "VNM"
    ]
    iso_codes = [
    'ATG', 'AIA', 'YEM', 'SEN', 'BWA', 'MDG', 'BEN', 'BIH', 'BLZ', 'BRB', 
    'CRI', 'DMA', 'GHA', 'GIN', 'GRD', 'HND', 'HUN', 'KAZ', 'KEN', 'KIR', 
    'KNA', 'LCA', 'MNG', 'MSR', 'MWI', 'NAM', 'NER', 'NGA', 'PAN', 'RWA', 
    'SLE', 'SLV', 'SSD', 'THA', 'TTO', 'UKR', 'UZB', 'VCT', 'VGB', 'ZAF', 
    'ZWE', 'BRA', 
    
    #'KHM', 'LAO', 'IDN', 'PHL', 'MYS', 'MMR', 'BGD', 'BRN'
    ]
    c["iso_codes"] = iso_codes
    iso = iso_codes[0]

    

    if "name" in c: iso = c["name"]
    c["iso_code"] = iso
    log_c = {
        key: val for key, val in c.items() 
        if (key is not None) 
        and ('url' not in key) 
        and ('dir' not in key)
        and ('file' not in key)
    }
    #logging.info(log_c)


    test(c, args.exp_name, args.dataset)
