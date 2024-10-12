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

from torchvision import transforms

import torch.nn as nn
import numpy as np
import pandas as pd

# Get device
cwd = os.path.dirname(os.getcwd())
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
logging.info(f"Device: {device}")

SEED = 42

imagenet_mean, imagenet_std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
test_transforms = []
for i in range(8):
    t = transforms.Compose(
            [
                
                #transforms.Resize(size),
                #transforms.CenterCrop(size),
                transforms.RandomRotation((i*45,i*45)),
                transforms.CenterCrop(500),
                transforms.ToTensor(),
                transforms.Normalize(imagenet_mean, imagenet_std),
            ]
        )
    test_transforms.append(t)

# def test(config, exp_name="all", dataset_name=None):

#     exp_name = f"{exp_name}"
#     exp_dir = os.path.join(cwd, c["exp_dir"], exp_name)

#     phases = ["train", "test"]
#     data, data_loader, classes = cnn_utils.load_dataset(config=c, phases=phases, name = dataset_name)

    

#     model, criterion, optimizer, scheduler = cnn_utils.load_model(
#         n_classes=len(classes),
#         model_type=c["model"],
#         pretrained=c["pretrained"],
#         scheduler_type=c["scheduler"],
#         optimizer_type=c["optimizer"],
#         label_smoothing=c["label_smoothing"],
#         lr=c["lr"],
#         momentum=c["momentum"],
#         gamma=c["gamma"],
#         step_size=c["step_size"],
#         patience=c["patience"],
#         dropout=c["dropout"],
#         device=device,
#     )

#     model_file = os.path.join(exp_dir, f"{exp_name}.pth")
#     model.load_state_dict(torch.load(model_file, map_location=device))
#     model = model.to(device)

#     # test_results, test_cm, test_preds = cnn_utils.evaluate(
#     #     data_loader["test"], classes, model, criterion, device, pos_label=1, wandb=wandb, logging=logging
#     # )
#     # test_preds.to_csv(os.path.join(exp_dir, f"{exp_name}.csv"), index=False)

#     # # Save results in experiment directory
#     # eval_utils._save_files(test_results, test_cm, exp_dir)

#     for iso_code in config["iso_codes"]:
#         try:
#             subresults_dir = os.path.join(exp_dir, iso_code + "_ensembling_max")
#             if not os.path.exists(subresults_dir):
#                     os.makedirs(subresults_dir)

#             #print(data["test"].dataset.columns)
#             subdata = data["test"].dataset[data["test"].dataset.iso == iso_code]
#             #print(f"{iso_code}: {len(subdata)}")

#             classes_dict = {config["pos_class"] : 1, config["neg_class"]: 0}
#             preds = []
#             for transform in test_transforms:
#                 dataset =  cnn_utils.SchoolDataset(
#                         subdata
#                         .sample(frac=1, random_state=SEED)
#                         .reset_index(drop=True),
#                         classes_dict,
#                         transform
#                 )


#                 data_loader =  torch.utils.data.DataLoader(
#                         dataset,
#                         batch_size=config["batch_size"],
#                         num_workers=config["n_workers"],
#                         shuffle=False,
#                         drop_last=False
#                 )

#                 if len(data_loader) == 0:
#                     continue

#                 test_results, test_cm, test_preds = cnn_utils.evaluate(
#                     data_loader, classes, model, criterion, device, pos_label=1, wandb=wandb, logging=logging
#                 )
#                 test_preds.loc[test_preds['y_preds'] == 0, 'y_probs'] = 1 - test_preds['y_probs']
#                 preds.append(test_preds)
#                 #test_preds.to_csv(os.path.join(subresults_dir, f"{iso_code}.csv"), index=False)

#             concatenated_probs = pd.concat([df["y_probs"] for df in preds], axis=1)
#             average_probs = concatenated_probs.max(axis=1)
#             new_preds = preds[0]
#             new_preds["y_probs"] = average_probs
#             new_preds["y_preds"] = (average_probs >= 0.5).astype(int)
#             new_preds.to_csv(os.path.join(subresults_dir, f"{iso_code}.csv"), index=False)

#             results = eval_utils.evaluate(new_preds["y_true"], new_preds["y_preds"], 1)
#             cm = eval_utils.get_confusion_matrix(new_preds["y_true"], new_preds["y_preds"], classes)         

#             # Save results in experiment directory
#             eval_utils._save_files(results, cm, subresults_dir)
#         except Exception as e:
#             print(f"error with code{iso_code}: {e}")

def test_all(config, exp_name="all", dataset_name=None):

    exp_name = f"{exp_name}"
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

    model_file = os.path.join(exp_dir, f"model.pth")
    if not os.path.exists(model_file):
         model_file = os.path.join(exp_dir, f"{exp_name}.pth")
    model.load_state_dict(torch.load(model_file, map_location=device))
    model = model.to(device)

    subresults_dir = os.path.join(exp_dir, f"test_ensembling_rotation_mean_{dataset_name}")
    if not os.path.exists(subresults_dir):
            os.makedirs(subresults_dir)

    #print(data["test"].dataset.columns)
    subdata = data["test"].dataset
    #print(f"{iso_code}: {len(subdata)}")

    classes_dict = {config["pos_class"] : 1, config["neg_class"]: 0}
    preds = []
    for transform in test_transforms:
        dataset =  cnn_utils.SchoolDataset(
                subdata
                .sample(frac=1, random_state=SEED)
                .reset_index(drop=True),
                classes_dict,
                transform
        )


        data_loader =  torch.utils.data.DataLoader(
                dataset,
                batch_size=config["batch_size"],
                num_workers=config["n_workers"],
                shuffle=False,
                drop_last=False
        )

        if len(data_loader) == 0:
            continue

        test_results, test_cm, test_preds = cnn_utils.evaluate(
            data_loader, classes, model, criterion, device, pos_label=1, wandb=wandb, logging=logging
        )
        test_preds.loc[test_preds['y_preds'] == 0, 'y_probs'] = 1 - test_preds['y_probs']
        preds.append(test_preds)
        #test_preds.to_csv(os.path.join(subresults_dir, f"{iso_code}.csv"), index=False)

    concatenated_probs = pd.concat([df["y_probs"] for df in preds], axis=1)
    average_probs = concatenated_probs.mean(axis=1)
    new_preds = preds[0]
    new_preds["y_probs"] = average_probs
    new_preds["y_preds"] = (average_probs >= 0.5).astype(int)
    new_preds.to_csv(os.path.join(subresults_dir, f"preds.csv"), index=False)

    results = eval_utils.evaluate(new_preds["y_true"], new_preds["y_preds"], 1)
    cm = eval_utils.get_confusion_matrix(new_preds["y_true"], new_preds["y_preds"], classes)         

    # Save results in experiment directory
    eval_utils._save_files(results, cm, subresults_dir)


if __name__ == "__main__":
    # Parser
    parser = argparse.ArgumentParser(description="Model Training")
    parser.add_argument("--cnn_config", help="Config file", default="convnext_small")
    parser.add_argument("--device", help="device", default="cuda:1")
    parser.add_argument('-e', "--exp_name", default="global_no_vietnam_500images_no_lowres_continuous_rotation_0-90_crop352_no_AMP_convnext_small/fine_tune_vietnam_large")
    parser.add_argument('--dataset', default="vietnam")
    args = parser.parse_args()

    device = torch.device(args.device)
    logging.info(f"config: {args.cnn_config}")
    logging.info(f"Args Device: {device}")
    logging.info(f"exp_name: {args.dataset}_{args.exp_name}")
    logging.info(f"dataset: {args.dataset}")

    # Load config
    config_file = os.path.join(cwd, "configs", "cnn_configs", args.cnn_config + ".yaml")
    c = config_utils.load_config(config_file)

    test_all(c, args.exp_name, args.dataset)
