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
import data_utils

import torch.nn as nn
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import models, transforms

# Get device
cwd = os.path.dirname(os.getcwd())
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
logging.info(f"Device: {device}")
SEED = 42

classes_dict = {"school" : 1, "non_school": 0}
imagenet_mean, imagenet_std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
train_transform = transforms.Compose(
    [
        #transforms.Resize(size),
        #transforms.RandomApply([transforms.RandomRotation((90, 90))], p=0.5),
        transforms.RandomRotation((0,360)),
        transforms.CenterCrop(500),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean, imagenet_std),
    ]
)
class SchoolDataset(Dataset):
    def __init__(self, dataset, classes, transform=None):
        """
        Custom dataset for Caribbean images.

        Args:
        - dataset (pandas.DataFrame): The dataset containing image information.
        - attribute (str): The column name specifying the attribute for classification.
        - classes (dict): A dictionary mapping attribute values to classes.
        - transform (callable, optional): Optional transformations to apply to the image. 
        Defaults to None.
        - prefix (str, optional): Prefix to append to file paths. Defaults to an empty string.
        """
        
        self.dataset = dataset
        self.transform = transform
        self.classes = classes

    def __getitem__(self, index):
        """
        Retrieves an item (image and label) from the dataset based on index.

        Args:
        - index (int): Index of the item to retrieve.

        Returns:
        - tuple: A tuple containing the transformed image (if transform is specified)
        and its label.
        """
        
        item = self.dataset.iloc[index]
        uid = ""
        filepath= item["filepath"]
        image = Image.open(filepath).convert("RGB")

        if self.transform:
            x = self.transform(image)

        y = self.classes[item["class"]]
        image.close()
        return x, y, uid

    def __len__(self):
        """
        Returns the length of the dataset.

        Returns:
        - int: Length of the dataset.
        """
        
        return len(self.dataset)

def main(c, exp_name="all", finetune=""):    
    # f = "/mnt/ssd1/agorup/school_mapping/inference_data/Anditi_filtered_schools_2-3857.csv"
    # data = pd.read_csv(f)
    # dest_dir = '/mnt/ssd1/agorup/school_mapping/satellite_images/anditi/large'

    cwd = os.path.dirname(os.getcwd())
    exp_name = f"{exp_name}_{c['config_name']}"
    exp_dir = os.path.join(cwd, c["exp_dir"], exp_name)
    if finetune != "":
        exp_dir = os.path.join(exp_dir, f"fine_tune_{finetune}")
    print(exp_dir)
    model_file = os.path.join(exp_dir, f"model.pth")
    if not os.path.exists(model_file):
        model_file = os.path.join(exp_dir, f"{exp_name}.pth")
    
    log_string = ""
    
    # Load dataset
    f = os.path.join(exp_dir, "anditi_school.csv")
    #f = "/home/agorup/school_mapping/exp/global_no_vietnam_500images_no_lowres_continuous_rotation_0-90_crop352_no_AMP_convnext_small/fine_tune_vietnam_large/anditi_school.csv"
    data = pd.read_csv(f)
    data = data[data["pred"] >= 0.5]
    dest_dir = '/mnt/ssd1/agorup/school_mapping/satellite_images/anditi/large'
    images_school = []
    for i in range(len(data)):
        image_file = f"{dest_dir}/{data.iloc[i]['image']}"
        images_school.append(image_file)

    phases = ["train", "test"]
    data, data_loader, classes = cnn_utils.load_dataset(config=c, phases=phases, name = "vietnam")
    data = data["train"].dataset

    # subdata = data
    # geoboundaries = data_utils._get_geoboundaries(c, "VNM", adm_level="ADM1")
    # geoboundaries = geoboundaries[["shapeName", "geometry"]].dropna(subset=["shapeName"])
    # geoboundaries = geoboundaries.to_crs(subdata.crs)
    # subdata = subdata.sjoin(geoboundaries, how="left", predicate="within")

    data = data[data['class']=="non_school"]
    data = data[data['clean']==0]
    data = data.sample(n=1*len(images_school))

    images_non_school = []
    for i, row in data.iterrows():
        image_file = f"/mnt/ssd1/agorup/school_mapping/satellite_images/large/VNM/non_school/{row['UID']}.jpeg"
        images_non_school.append(image_file)

    df = pd.DataFrame(columns=["filepath","class"])
    for img in images_school:
        row = {"filepath":img, "class":"school"}
        df.loc[len(df)] = row

    for img in images_non_school:
        row = {"filepath":img, "class":"non_school"}
        df.loc[len(df)] = row

    dataset = SchoolDataset(df, classes_dict, train_transform)
    data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=c["batch_size"],
            num_workers=c["n_workers"],
            shuffle=True,
            drop_last=False
        )
    
    exp_dir = os.path.join(exp_dir, f"fine_tune_anditi")
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
   

    # Load model, optimizer, and scheduler
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

    n_epochs = 13
    since = time.time()
    best_score = -1

    for epoch in range(1, n_epochs + 1):
        logging.info("\nEpoch {}/{}".format(epoch, n_epochs))

        # Train model
        train_results = cnn_utils.train(
            data_loader,
            model,
            criterion,
            optimizer,
            device,
            pos_label=1,
            wandb=wandb,
            logging=logging
        )

        log_string += f"epoch {epoch}:  train F1 = {train_results['f1_score']}, "

        # Terminate if learning rate becomes too low
        learning_rate = optimizer.param_groups[0]["lr"]
        if learning_rate < 1e-10:
            break

    model_file = os.path.join(exp_dir, "model.pth")
    torch.save(model.state_dict(), model_file)

    # Terminate trackers
    time_elapsed = time.time() - since
    logging.info(
        "Training complete in {:.0f}m {:.0f}s".format(
            time_elapsed // 60, time_elapsed % 60
        )
    )

    log_string += "Training complete in {:.0f}m {:.0f}s\n".format(
            time_elapsed // 60, time_elapsed % 60
        )
    log_file = os.path.join(exp_dir, "log.txt")
    f = open(log_file, "w")
    f.write(log_string)
    f.close()

if __name__ == "__main__":
    # Parser
    parser = argparse.ArgumentParser(description="Model Training")
    parser.add_argument("--cnn_config", help="Config file", default="convnext_small")
    parser.add_argument('-d', "--device", help="device", default="cuda:0")
    parser.add_argument('-e', "--exp_name", default="global_no_vietnam_500images_no_lowres_continuous_rotation_0-90_crop352_no_AMP")
    parser.add_argument("-f", '--fine_tune_dataset', default="vietnam_uninhabited")
    args = parser.parse_args()

    device = torch.device(args.device)

    # Load config
    config_file = os.path.join(cwd, "configs", "cnn_configs", args.cnn_config + ".yaml")
    c = config_utils.load_config(config_file)

    main(c, args.exp_name, args.fine_tune_dataset)