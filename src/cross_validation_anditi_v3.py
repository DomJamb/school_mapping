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
from tqdm import tqdm

import torch.nn as nn
import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal
from sklearn.metrics import precision_recall_curve, auc
from PIL import Image
from torch.utils.data import Dataset
from torchvision import models, transforms
import copy
import random
import numpy
import torch.nn.functional as nnf
import matplotlib.pyplot as plt

# Get device
cwd = os.path.dirname(os.getcwd())
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
logging.info(f"Device: {device}")
SEED = 40
random.seed(SEED)
numpy.random.seed(SEED)

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

rotation_transforms = []
tencrop_transforms = []
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
    rotation_transforms.append(t)
t = transforms.Compose(
        [
            
            #transforms.Resize(size),
            #transforms.CenterCrop(size),
            transforms.TenCrop(size=500),
        ]
    )
tencrop_transforms.append(t)

t_tensor = transforms.Compose(
    [
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
        uid = item["UID"]
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
    
def calculate_euclidean_distance(point_df, point):
    point_df = np.array([point_df.x, point_df.y])

    # Calculate Euclidean distance
    dist =  ((point_df[0] - point[0]) ** 2 + (point_df[1] - point[1]) ** 2) ** 0.5

    return dist

def calculate_bivar_gaussian_pdf(point_df, mean, cov_matrix):
    point_df = np.array([point_df.x, point_df.y])
    
    # Calculate bivariate gaussian probability density
    prob_density = multivariate_normal.pdf(point_df, mean, cov_matrix)

    return prob_density

def sample_non_schools(cluster_rows, data_ns, sampling_mode='inverse', OHEM=False):
    dataset_ns = data_ns.copy()

    # Calculate centroid
    centroid = (cluster_rows["lon"].mean(), cluster_rows["lat"].mean())

    if sampling_mode == 'inverse':
        # Calculate distance from centroid
        dataset_ns["cluster_dist"] = dataset_ns["geometry"].apply(lambda row: calculate_euclidean_distance(row, centroid))

        # Calculate probability to belong to centroid (1 / d)
        dataset_ns["cluster_prob"] = 1 / (dataset_ns["cluster_dist"] + 1e-10)
        dataset_ns["cluster_prob"] = dataset_ns["cluster_prob"] / dataset_ns["cluster_prob"].sum()
    elif sampling_mode == 'gaussian':
        # Calculate variance
        var = (cluster_rows["lon"].var(), cluster_rows["lat"].var())

        # Calculate covariance matrix
        cov_matrix = np.array([
            [2 * var[0], 0],
            [0, 2 * var[1]]
        ])

        # Calculate probability to belong to centroid (bivariate Gaussian)
        dataset_ns["cluster_prob"] = dataset_ns["geometry"].apply(lambda row: calculate_bivar_gaussian_pdf(row, centroid, cov_matrix))
        dataset_ns["cluster_prob"] = dataset_ns["cluster_prob"] / dataset_ns["cluster_prob"].sum()

    if OHEM:
        # Calculate multiplicative probability
        dataset_ns["school_x_cluster"] = dataset_ns["y_probs_pos"] * dataset_ns["cluster_prob"]

        # Calculate sizes
        size1 = int(len(cluster_rows) / 2 + 0.5)
        size2 = int(len(cluster_rows) / 2)

        # Choose nonschools based on multiplicative probability
        cluster_ns_indices1 = np.random.choice(dataset_ns.index, size=size1, replace=False, p=dataset_ns["school_x_cluster"])
        dataset_ns = dataset_ns.drop(cluster_ns_indices1)

        # Choose nonschools based on cluster probability
        dataset_ns["cluster_prob"] = dataset_ns["cluster_prob"] / dataset_ns["cluster_prob"].sum()
        cluster_ns_indices2 = np.random.choice(dataset_ns.index, size=size2, replace=False, p=dataset_ns["cluster_prob"])

        cluster_ns_indices = np.concatenate((cluster_ns_indices1, cluster_ns_indices2))
    else:
        # Choose nonschools for cluster based on probability
        cluster_ns_indices = np.random.choice(dataset_ns.index, size=len(cluster_rows), replace=False, p=dataset_ns["cluster_prob"])

    cluster_ns = dataset_ns.loc[cluster_ns_indices]

    return cluster_ns

def plot_pr(preds_path, save_path):
    # Train PR curve
    plt.figure(figsize=(8,6))
    plt.title('Precision-Recall Curve, Train')
    plt.xlabel('Recall')
    plt.ylabel('Precision')

    for i, file_name in enumerate(["train_preds1.csv", "train_preds2.csv"]):
        f = os.path.join(preds_path, file_name)
        results = pd.read_csv(f)

        # Set positive class probs
        results["y_probs_pos"] = results.apply(lambda row: row["y_probs"] if row["y_preds"] == 1 else 1 - row["y_probs"], axis=1)

        precision, recall, _ = precision_recall_curve(results["y_true"], results["y_probs_pos"], pos_label=1)
        auc_score = auc(recall, precision)

        plt.plot(recall, precision, label=f'{"North" if i == 0 else "South"} (AUC = {auc_score:.2f})')

    plt.legend()
    plt.savefig(os.path.join(save_path, "PR_train.png"))

    # Val PR curve
    plt.figure(figsize=(8,6))
    plt.title('Precision-Recall Curve, Val')
    plt.xlabel('Recall')
    plt.ylabel('Precision')

    for i, file_name in enumerate(["val_preds1.csv", "val_preds2.csv"]):
        f = os.path.join(preds_path, file_name)
        results = pd.read_csv(f)

        # Set positive class probs
        results["y_probs_pos"] = results.apply(lambda row: row["y_probs"] if row["y_preds"] == 1 else 1 - row["y_probs"], axis=1)

        precision, recall, _ = precision_recall_curve(results["y_true"], results["y_probs_pos"], pos_label=1)
        auc_score = auc(recall, precision)

        plt.plot(recall, precision, label=f'{"North -> South" if i == 0 else "South -> North"}  (AUC = {auc_score:.2f})')

    plt.legend()
    plt.savefig(os.path.join(save_path, "PR_val.png"))

def main(c, exp_name="all", sampling="inverse", OHEM=False):    
    # f = "/mnt/sdb/agorup/school_mapping/inference_data/Anditi_filtered_schools_2-3857.csv"
    # data = pd.read_csv(f)
    # dest_dir = '/mnt/sdb/agorup/school_mapping/satellite_images/anditi/large'

    cwd = os.path.dirname(os.getcwd())
    exp_dir = os.path.join(cwd, c["exp_dir"], exp_name)
    print(exp_dir)
    model_file = os.path.join(exp_dir, f"model.pth")
    if not os.path.exists(model_file):
        model_file = os.path.join(exp_dir, f"{exp_name}.pth")

    crossval_dir = os.path.join(cwd, c["exp_dir"], "cross_validation_anditi", "dynamic", sampling)
    if not os.path.exists(crossval_dir):
        os.makedirs(crossval_dir)

    data_dir = os.path.join(crossval_dir, f"data")
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    results_string = ""
    
    # Load dataset
    f = os.path.join(exp_dir, "anditi_school.csv")
    data = pd.read_csv(f)
    n_schools = len(data)
    
    assert n_schools == 1003

    with open(os.path.join(exp_dir, "anditi_cluster_1.txt"), "r") as f:
        cluster_1 = [line.strip() for line in f.readlines() if len(line.strip()) > 0]
    with open(os.path.join(exp_dir, "anditi_cluster_2.txt"), "r") as f:
        cluster_2 = [line.strip() for line in f.readlines() if len(line.strip()) > 0]   

    # Filter dataframe for each cluster
    cluster_1_rows = data.loc[data["image"].isin([f"{img_id}.jpeg" for img_id in cluster_1])]
    cluster_2_rows = data.loc[data["image"].isin([f"{img_id}.jpeg" for img_id in cluster_2])]
    
    dest_dir = '/mnt/sdb/agorup/school_mapping/satellite_images/anditi/large'

    images_school_1 = []
    for img in cluster_1:
        image_file = f"{dest_dir}/{img}.jpeg"
        images_school_1.append(image_file)

    images_school_2 = []
    for img in cluster_2:
        image_file = f"{dest_dir}/{img}.jpeg"
        images_school_2.append(image_file)

    phases = ["train", "test"]
    data, data_loader, classes = cnn_utils.load_dataset(config=c, phases=phases, name = "vietnam")
    data = data["train"].dataset
    data = data[data['class']=="non_school"]
    data = data[data['clean']==0]
    data = data.to_crs('EPSG:3857')

    if OHEM:
        df_all = pd.DataFrame(columns=["filepath","UID","class"])
        for img in [*cluster_1, *cluster_2]:
            image_file = f"{dest_dir}/{img}.jpeg"
            uid = img
            row = {"filepath": image_file, "UID": uid, "class": "school"}
            df_all.loc[len(df_all)] = row
        for i, row in data.iterrows():
            image_file = f"/mnt/sdb/agorup/school_mapping/satellite_images/large/VNM/non_school/{row['UID']}.jpeg"
            uid = row['UID']
            row = {"filepath": image_file, "UID": uid, "class": "non_school"}
            df_all.loc[len(df_all)] = row

        dataset_all = SchoolDataset(df_all, classes_dict, train_transform)
        data_loader_all = torch.utils.data.DataLoader(
                dataset_all,
                batch_size=c["batch_size"],
                num_workers=c["n_workers"],
                shuffle=False,
                drop_last=False
            )

    data_ns_c1 = sample_non_schools(cluster_1_rows, data, sampling)
    data_ns_c1.to_csv(os.path.join(data_dir, "ep1_non_schools_cluster_1.csv"))

    images_non_school_1 = []
    for i, row in data_ns_c1.iterrows():
        image_file = f"/mnt/sdb/agorup/school_mapping/satellite_images/large/VNM/non_school/{row['UID']}.jpeg"
        images_non_school_1.append(image_file)

    df1 = pd.DataFrame(columns=["filepath","UID","class"])
    for i in range(len(images_school_1)):
        img = images_school_1[i]
        uid = img.split("/")[-1].replace(".jpeg", "")
        row = {"filepath":img, "UID": uid, "class":"school"}
        df1.loc[len(df1)] = row
    for i in range(len(images_non_school_1)):
        img = images_non_school_1[i]
        uid = img.split("/")[-1].replace(".jpeg", "")
        row = {"filepath":img, "UID": uid, "class":"non_school"}
        df1.loc[len(df1)] = row
    df1.to_csv(os.path.join(crossval_dir, "df1.csv"), index=False)

    dataset1 = SchoolDataset(df1, classes_dict, train_transform)
    data_loader1 = torch.utils.data.DataLoader(
            dataset1,
            batch_size=c["batch_size"],
            num_workers=c["n_workers"],
            shuffle=True,
            drop_last=False
        )

    data_ns_c2 = sample_non_schools(cluster_2_rows, data, sampling)
    data_ns_c2.to_csv(os.path.join(data_dir, "ep1_non_schools_cluster_2.csv"))

    images_non_school_2 = []
    for i, row in data_ns_c2.iterrows():
        image_file = f"/mnt/sdb/agorup/school_mapping/satellite_images/large/VNM/non_school/{row['UID']}.jpeg"
        images_non_school_2.append(image_file)

    df2 = pd.DataFrame(columns=["filepath","UID","class"])
    for i in range(len(images_school_2)):
        img = images_school_2[i]
        uid = img.split("/")[-1].replace(".jpeg", "")
        row = {"filepath":img, "UID": uid, "class":"school"}
        df2.loc[len(df2)] = row
    for i in range(len(images_non_school_2)):
        img = images_non_school_2[i]
        uid = img.split("/")[-1].replace(".jpeg", "")
        row = {"filepath":img, "UID": uid, "class":"non_school"}
        df2.loc[len(df2)] = row
    df2.to_csv(os.path.join(crossval_dir, "df2.csv"), index=False)
    
    dataset2 = SchoolDataset(df2, classes_dict, train_transform)
    data_loader2 = torch.utils.data.DataLoader(
            dataset2,
            batch_size=c["batch_size"],
            num_workers=c["n_workers"],
            shuffle=True,
            drop_last=False
        )
   
    classes = ['school', 'non_school']
    n_epochs = 13
    since = time.time()

    # FIRST PASS
    log_string_1 = ""
    model1, criterion, optimizer, scheduler = cnn_utils.load_model(
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
    model1.load_state_dict(torch.load(model_file, map_location=device))
    model1 = model1.to(device)

    for epoch in range(1, n_epochs + 1):
        if epoch > 1:
            # Delete previous dataset and dataloader
            del dataset1
            del data_loader1
            torch.cuda.empty_cache()

            # Get model predictions for all non-schools (if OHEM sampling)
            if OHEM:
                _, _, preds = cnn_utils.evaluate(
                    data_loader_all, 
                    classes, 
                    model1, 
                    criterion, 
                    device, 
                    pos_label=1,
                    wandb=wandb, 
                    logging=logging
                )

                # Filter so that only non-schools are in dataframe
                preds = preds[preds["y_true"] == 0]

                # Set positive class probs
                preds["y_probs_pos"] = preds.apply(lambda row: row["y_probs"] if row["y_preds"] == 1 else 1 - row["y_probs"], axis=1)

                # Merge dataframes
                data_copy = data.merge(preds[["UID", "y_probs_pos"]], on="UID", how="left")

                # Normalize probability
                data_copy["y_probs_pos"] = data_copy["y_probs_pos"].fillna(0)
                data_copy["y_probs_pos"] = data_copy["y_probs_pos"] / data_copy["y_probs_pos"].sum()
            else:
                data_copy = data.copy()

            # Sample non-schools
            data_ns_c1 = sample_non_schools(cluster_1_rows, data_copy, sampling, OHEM)
            data_ns_c1.to_csv(os.path.join(data_dir, f"ep{epoch}_non_schools_cluster_1.csv"))

            # Find non-school files
            images_non_school_1 = []
            for i, row in data_ns_c1.iterrows():
                image_file = f"/mnt/sdb/agorup/school_mapping/satellite_images/large/VNM/non_school/{row['UID']}.jpeg"
                images_non_school_1.append(image_file)

            # Create dataframe
            df1 = pd.DataFrame(columns=["filepath","UID","class"])
            for i in range(len(images_school_1)):
                img = images_school_1[i]
                uid = img.split("/")[-1].replace(".jpeg", "")
                row = {"filepath":img, "UID": uid, "class":"school"}
                df1.loc[len(df1)] = row
            for i in range(len(images_non_school_1)):
                img = images_non_school_1[i]
                uid = img.split("/")[-1].replace(".jpeg", "")
                row = {"filepath":img, "UID": uid, "class":"non_school"}
                df1.loc[len(df1)] = row
            df1.to_csv(os.path.join(crossval_dir, "df1.csv"), index=False)

            # Create dataset and data loader
            dataset1 = SchoolDataset(df1, classes_dict, train_transform)
            data_loader1 = torch.utils.data.DataLoader(
                    dataset1,
                    batch_size=c["batch_size"],
                    num_workers=c["n_workers"],
                    shuffle=True,
                    drop_last=False
                )
    
        logging.info("\nModel 1, Epoch {}/{}".format(epoch, n_epochs))

        # Train model
        train_results = cnn_utils.train(
            data_loader1,
            model1,
            criterion,
            optimizer,
            device,
            pos_label=1,
            wandb=wandb,
            logging=logging
        )
        log_string_1 += "Model 1, Epoch {}/{}: train F1 = {}\n".format(epoch, n_epochs, train_results['f1_score'])

        # Terminate if learning rate becomes too low
        learning_rate = optimizer.param_groups[0]["lr"]
        if learning_rate < 1e-10:
            break

    train_results1, train_cm, train_preds = cnn_utils.evaluate(
            data_loader1, 
            classes, 
            model1, 
            criterion, 
            device, 
            pos_label=1,
            wandb=wandb, 
            logging=logging
        )

    val_results1, val_cm, val_preds = cnn_utils.evaluate(
            data_loader2, 
            classes, 
            model1, 
            criterion, 
            device, 
            pos_label=1,
            wandb=wandb, 
            logging=logging
        )
    
    train_preds.to_csv(os.path.join(crossval_dir, "train_preds1.csv"))
    val_preds.to_csv(os.path.join(crossval_dir, "val_preds1.csv"))

    #model_file = os.path.join(exp_dir, "model1.pth")
    torch.save(model1.state_dict(), os.path.join(crossval_dir, f"crossval_model_1.pth"))

    """
    model1_rotation_results = np.zeros(4)
    model1_tencrop_results = np.zeros(4)
    for i in tqdm(range(len(df2))):
        row = df2.iloc[i]
        image_file = row['filepath']
        img = Image.open(image_file).convert("RGB")
        img_tensors = []
        for j in range(len(rotation_transforms)):
            img2 = rotation_transforms[j](img)
            img2 = img2.unsqueeze(0).to(device)
            img_tensors.append(img2)
        img_tensors = torch.cat(img_tensors, dim=0)
        out = model1(img_tensors)
        soft_outputs = nnf.softmax(out, dim=1)
        positive_probs = soft_outputs[:, 1]
        positive_prob = torch.mean(positive_probs).item()
        pred_rotation = int(positive_prob > 0.5)
        if pred_rotation == 1 and row['class'] == 'school':
            model1_rotation_results[0] += 1
        elif pred_rotation == 0 and row['class'] == 'school':
            model1_rotation_results[1] += 1
        elif pred_rotation == 1 and row['class'] == 'non_school':
            model1_rotation_results[2] += 1
        elif pred_rotation == 0 and row['class'] == 'non_school':
            model1_rotation_results[3] += 1

        img_tensors = []
        for j in range(len(tencrop_transforms)):
            img2 = tencrop_transforms[j](img)
            for img3 in img2:
                img3 = t_tensor(img3)
                img3 = img3.unsqueeze(0).to(device)
                img_tensors.append(img3)
        img_tensors = torch.cat(img_tensors, dim=0)
        out = model1(img_tensors)
        soft_outputs = nnf.softmax(out, dim=1)
        positive_probs = soft_outputs[:, 1]
        positive_prob = torch.max(positive_probs).item()
        pred_tencrop = int(positive_prob > 0.5)
        if pred_tencrop == 1 and row['class'] == 'school':
            model1_tencrop_results[0] += 1
        elif pred_tencrop == 0 and row['class'] == 'school':
            model1_tencrop_results[1] += 1
        elif pred_tencrop == 1 and row['class'] == 'non_school':
            model1_tencrop_results[2] += 1
        elif pred_tencrop == 0 and row['class'] == 'non_school':
            model1_tencrop_results[3] += 1
    """

    # Free up CUDA memory
    del model1
    del optimizer
    del scheduler
    torch.cuda.empty_cache()
    
    # SECOND PASS
    log_string_2 = ""
    model2, criterion, optimizer, scheduler = cnn_utils.load_model(
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
    model2.load_state_dict(torch.load(model_file, map_location=device))
    model2 = model2.to(device)
    for epoch in range(1, n_epochs + 1):
        if epoch > 1:
            # Delete previous dataset and dataloader
            del dataset2
            del data_loader2
            torch.cuda.empty_cache()

            # Get model predictions for all non-schools (if OHEM sampling)
            if OHEM:
                _, _, preds = cnn_utils.evaluate(
                    data_loader_all, 
                    classes, 
                    model2, 
                    criterion, 
                    device, 
                    pos_label=1,
                    wandb=wandb, 
                    logging=logging
                )

                # Filter so that only non-schools are in dataframe
                preds = preds[preds["y_true"] == 0]

                # Set positive class probs
                preds["y_probs_pos"] = preds.apply(lambda row: row["y_probs"] if row["y_preds"] == 1 else 1 - row["y_probs"], axis=1)

                # Merge dataframes
                data_copy = data.merge(preds[["UID", "y_probs_pos"]], on="UID", how="left")

                # Normalize probability
                data_copy["y_probs_pos"] = data_copy["y_probs_pos"].fillna(0)
                data_copy["y_probs_pos"] = data_copy["y_probs_pos"] / data_copy["y_probs_pos"].sum()
            else:
                data_copy = data.copy()

            data_ns_c2 = sample_non_schools(cluster_2_rows, data_copy, sampling, OHEM)
            data_ns_c2.to_csv(os.path.join(data_dir, f"ep{epoch}_non_schools_cluster_2.csv"))

            images_non_school_2 = []
            for i, row in data_ns_c2.iterrows():
                image_file = f"/mnt/sdb/agorup/school_mapping/satellite_images/large/VNM/non_school/{row['UID']}.jpeg"
                images_non_school_2.append(image_file)

            df2 = pd.DataFrame(columns=["filepath","UID","class"])
            for i in range(len(images_school_2)):
                img = images_school_2[i]
                uid = img.split("/")[-1].replace(".jpeg", "")
                row = {"filepath":img, "UID": uid, "class":"school"}
                df2.loc[len(df2)] = row
            for i in range(len(images_non_school_2)):
                img = images_non_school_2[i]
                uid = img.split("/")[-1].replace(".jpeg", "")
                row = {"filepath":img, "UID": uid, "class":"non_school"}
                df2.loc[len(df2)] = row
            df2.to_csv(os.path.join(crossval_dir, "df2.csv"), index=False)
            
            dataset2 = SchoolDataset(df2, classes_dict, train_transform)
            data_loader2 = torch.utils.data.DataLoader(
                    dataset2,
                    batch_size=c["batch_size"],
                    num_workers=c["n_workers"],
                    shuffle=True,
                    drop_last=False
                )
    
        logging.info("\nModel 2, Epoch {}/{}".format(epoch, n_epochs))

        # Train model
        train_results = cnn_utils.train(
            data_loader2,
            model2,
            criterion,
            optimizer,
            device,
            pos_label=1,
            wandb=wandb,
            logging=logging
        )
        log_string_2 += "Model 2, Epoch {}/{}: train F1 = {}\n".format(epoch, n_epochs, train_results['f1_score'])

        # Terminate if learning rate becomes too low
        learning_rate = optimizer.param_groups[0]["lr"]
        if learning_rate < 1e-10:
            break

    train_results2, train_cm, train_preds = cnn_utils.evaluate(
            data_loader2, 
            classes, 
            model2, 
            criterion, 
            device, 
            pos_label=1,
            wandb=wandb, 
            logging=logging
        )

    val_results2, val_cm, val_preds = cnn_utils.evaluate(
            data_loader1, 
            classes, 
            model2, 
            criterion, 
            device, 
            pos_label=1,
            wandb=wandb, 
            logging=logging
        )
    
    train_preds.to_csv(os.path.join(crossval_dir, "train_preds2.csv"))
    val_preds.to_csv(os.path.join(crossval_dir, "val_preds2.csv"))

    #model_file = os.path.join(exp_dir, "model2.pth")
    torch.save(model2.state_dict(), os.path.join(crossval_dir, f"crossval_model_2.pth"))

    """
    model2_rotation_results = np.zeros(4)
    model2_tencrop_results = np.zeros(4)
    for i in tqdm(range(len(df1))):
        row = df1.iloc[i]
        image_file = row['filepath']
        img = Image.open(image_file).convert("RGB")
        img_tensors = []
        for j in range(len(rotation_transforms)):
            img2 = rotation_transforms[j](img)
            img2 = img2.unsqueeze(0).to(device)
            img_tensors.append(img2)
        img_tensors = torch.cat(img_tensors, dim=0)
        out = model2(img_tensors)
        soft_outputs = nnf.softmax(out, dim=1)
        positive_probs = soft_outputs[:, 1]
        positive_prob = torch.mean(positive_probs).item()
        pred_rotation = int(positive_prob > 0.5)
        if pred_rotation == 1 and row['class'] == 'school':
            model2_rotation_results[0] += 1
        elif pred_rotation == 0 and row['class'] == 'school':
            model2_rotation_results[1] += 1
        elif pred_rotation == 1 and row['class'] == 'non_school':
            model2_rotation_results[2] += 1
        elif pred_rotation == 0 and row['class'] == 'non_school':
            model2_rotation_results[3] += 1

        img_tensors = []
        for j in range(len(tencrop_transforms)):
            img2 = tencrop_transforms[j](img)
            for img3 in img2:
                img3 = t_tensor(img3)
                img3 = img3.unsqueeze(0).to(device)
                img_tensors.append(img3)
        img_tensors = torch.cat(img_tensors, dim=0)
        out = model2(img_tensors)
        soft_outputs = nnf.softmax(out, dim=1)
        positive_probs = soft_outputs[:, 1]
        positive_prob = torch.max(positive_probs).item()
        pred_tencrop = int(positive_prob > 0.5)
        if pred_tencrop == 1 and row['class'] == 'school':
            model2_tencrop_results[0] += 1
        elif pred_tencrop == 0 and row['class'] == 'school':
            model2_tencrop_results[1] += 1
        elif pred_tencrop == 1 and row['class'] == 'non_school':
            model2_tencrop_results[2] += 1
        elif pred_tencrop == 0 and row['class'] == 'non_school':
            model2_tencrop_results[3] += 1
    """

    # Free up CUDA memory
    del model2
    del optimizer
    del scheduler
    torch.cuda.empty_cache()
    
    f1_avg = (val_results1["f1_score"] + val_results2["f1_score"]) / 2
    precision_avg = (val_results1["precision_score"] + val_results2["precision_score"]) / 2
    recall_avg = (val_results1["recall_score"] + val_results2["recall_score"]) / 2

    """
    recall_rotation_avg = ((model1_rotation_results[0] / (model1_rotation_results[0] + model1_rotation_results[1])) + 
                           (model2_rotation_results[0] / (model2_rotation_results[0] + model2_rotation_results[1]))) / 2
    precision_rotation_avg = ((model1_rotation_results[0] / (model1_rotation_results[0] + model1_rotation_results[2])) + 
                           (model2_rotation_results[0] / (model2_rotation_results[0] + model2_rotation_results[2]))) / 2
    f1_rotation_avg = (2 * recall_rotation_avg * precision_rotation_avg) / (recall_rotation_avg + precision_rotation_avg)

    recall_tencrop_avg = ((model1_tencrop_results[0] / (model1_tencrop_results[0] + model1_tencrop_results[1])) + 
                           (model2_tencrop_results[0] / (model2_tencrop_results[0] + model2_tencrop_results[1]))) / 2
    precision_tencrop_avg = ((model1_tencrop_results[0] / (model1_tencrop_results[0] + model1_tencrop_results[2])) + 
                           (model2_tencrop_results[0] / (model2_tencrop_results[0] + model2_tencrop_results[2]))) / 2
    f1_tencrop_avg = (2 * recall_tencrop_avg * precision_tencrop_avg) / (recall_tencrop_avg + precision_tencrop_avg)
    """
    
    results_string = ""

    results_string += f"NORTH -> SOUTH, TRAIN F1: {train_results1['f1_score']}\n"
    results_string += f"NORTH -> SOUTH, TRAIN PRECISION: {train_results1['precision_score']}\n"
    results_string += f"NORTH -> SOUTH, TRAIN RECALL: {train_results1['recall_score']}\n"
    results_string += "\n"

    results_string += f"NORTH -> SOUTH, VAL F1: {val_results1['f1_score']}\n"
    results_string += f"NORTH -> SOUTH, VAL PRECISION: {val_results1['precision_score']}\n"
    results_string += f"NORTH -> SOUTH, VAL RECALL: {val_results1['recall_score']}\n"
    results_string += "\n"

    results_string += f"SOUTH -> NORTH, TRAIN F1: {train_results2['f1_score']}\n"
    results_string += f"SOUTH -> NORTH, TRAIN PRECISION: {train_results2['precision_score']}\n"
    results_string += f"SOUTH -> NORTH, TRAIN RECALL: {train_results2['recall_score']}\n"
    results_string += "\n"

    results_string += f"SOUTH -> NORTH, VAL F1: {val_results2['f1_score']}\n"
    results_string += f"SOUTH -> NORTH, VAL PRECISION: {val_results2['precision_score']}\n"
    results_string += f"SOUTH -> NORTH, VAL RECALL: {val_results2['recall_score']}\n"
    results_string += "\n"

    results_string += f"AVG F1: {f1_avg}\n"
    results_string += f"AVG PRECISION: {precision_avg}\n"
    results_string += f"AVG RECALL: {recall_avg}\n"
    results_string += "\n"

    """
    results_string += f"AVG F1 ROTATION: {f1_rotation_avg}\n"
    results_string += f"AVG PRECISION ROTATION: {precision_rotation_avg}\n"
    results_string += f"AVG RECALL ROTATION: {recall_rotation_avg}\n"
    results_string += "\n"
    results_string += f"AVG F1 TENCROP: {f1_tencrop_avg}\n"
    results_string += f"AVG PRECISION TENCROP: {precision_tencrop_avg}\n"
    results_string += f"AVG RECALL TENCROP: {recall_tencrop_avg}\n"
    results_string += "\n"
    results_string += f"model1_rotation_results: {model1_rotation_results}\n"
    results_string += f"model1_tencrop_results: {model1_tencrop_results}\n"
    results_string += f"model2_rotation_results: {model2_rotation_results}\n"
    results_string += f"model2_tencrop_results: {model2_tencrop_results}"
    """

    print(results_string)

    f = open(os.path.join(crossval_dir, "log_1.txt"), "w")
    f.write(log_string_1)
    f.close()
    f = open(os.path.join(crossval_dir, "log_2.txt"), "w")
    f.write(log_string_2)
    f.close()
    f = open(os.path.join(crossval_dir, "results.txt"), "w")
    f.write(results_string)
    f.close()

    plot_pr(crossval_dir, crossval_dir)
    
    # Terminate trackers
    time_elapsed = time.time() - since
    logging.info(
        "Training complete in {:.0f}m {:.0f}s".format(
            time_elapsed // 60, time_elapsed % 60
        )
    )



if __name__ == "__main__":
    # Parser
    parser = argparse.ArgumentParser(description="Model Training")
    parser.add_argument("--sampling", help="Chosen non-school sampling method", default="inverse")
    parser.add_argument("--OHEM", help="Choose non-schools based on hard examples", action="store_true") 
    parser.add_argument("--cnn_config", help="Config file", default="convnext_small")
    parser.add_argument('-d', "--device", help="device", default="cuda:0")
    parser.add_argument('-e', "--exp_name", default="global_no_vietnam_500images_no_lowres_continuous_rotation_0-90_crop352_no_AMP_convnext_small/fine_tune_vietnam_large")
    args = parser.parse_args()

    device = torch.device(args.device)

    # Load config
    config_file = os.path.join(cwd, "configs", "cnn_configs", args.cnn_config + ".yaml")
    c = config_utils.load_config(config_file)

    main(c, args.exp_name, args.sampling, args.OHEM)