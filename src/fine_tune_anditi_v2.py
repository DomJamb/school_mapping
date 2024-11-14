import os
import time
import random
import argparse
import torch

import sys
sys.path.insert(0, "../utils/")
import config_utils
import cnn_utils
import wandb
import logging

import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal
from sklearn.metrics import precision_recall_curve, auc
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import random
import numpy
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
        transforms.RandomRotation((0,360)),
        transforms.CenterCrop(500),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean, imagenet_std),
    ]
)

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

def sample_non_schools(cluster_1_rows, cluster_2_rows, dataset_ns, sampling_mode='inverse'):
    # Calculate centroids
    centroid1 = (cluster_1_rows["lon"].mean(), cluster_1_rows["lat"].mean())
    centroid2 = (cluster_2_rows["lon"].mean(), cluster_2_rows["lat"].mean())

    print(f'Sampling non-schools using {sampling_mode} method')

    if sampling_mode == 'inverse':
        # Calculate distance from centroids
        dataset_ns["c1_dist"] = dataset_ns["geometry"].apply(lambda row: calculate_euclidean_distance(row, centroid1))
        dataset_ns["c2_dist"] = dataset_ns["geometry"].apply(lambda row: calculate_euclidean_distance(row, centroid2))

        ### CENTROID 1
        # Calculate probability to belong to centroid 1 (1 / d1)
        dataset_ns["c1_prob"] = 1 / (dataset_ns["c1_dist"] + 1e-10)
        dataset_ns["c1_prob"] = dataset_ns["c1_prob"] / dataset_ns["c1_prob"].sum()

        # Choose nonschools for cluster 1 based on probability 1
        cluster1_ns_indices = np.random.choice(dataset_ns.index, size=len(cluster_1_rows), replace=False, p=dataset_ns["c1_prob"])
        cluster1_ns = dataset_ns.loc[cluster1_ns_indices]

        # Remove sampled nonschools
        dataset_ns = dataset_ns.drop(cluster1_ns_indices)

        ### CENTROID 2
        # Calculate probability to belong to centroid 2 (1 / d2)
        dataset_ns["c2_prob"] = 1 / (dataset_ns["c2_dist"] + 1e-10)
        dataset_ns["c2_prob"] = dataset_ns["c2_prob"] / dataset_ns["c2_prob"].sum()

        # Choose nonschools for cluster 2 based on probability 2
        cluster2_ns_indices = np.random.choice(dataset_ns.index, size=len(cluster_2_rows), replace=False, p=dataset_ns["c2_prob"])
        cluster2_ns = dataset_ns.loc[cluster2_ns_indices]
    elif sampling_mode == 'gaussian':
        # Calculate variance
        var1 = (cluster_1_rows["lon"].var(), cluster_1_rows["lat"].var())
        var2 = (cluster_2_rows["lon"].var(), cluster_2_rows["lat"].var())

        # Calculate covariance matrices
        cov_matrix1 = np.array([
            [2 * var1[0], 0],
            [0, 2 * var1[1]]
        ])

        cov_matrix2 = np.array([
            [2 * var2[0], 0],
            [0, 2 * var2[1]]
        ])

        ### CENTROID 1
        # Calculate probability to belong to centroid 1 (bivariate Gaussian)
        dataset_ns["c1_prob"] = dataset_ns["geometry"].apply(lambda row: calculate_bivar_gaussian_pdf(row, centroid1, cov_matrix1))
        dataset_ns["c1_prob"] = dataset_ns["c1_prob"] / dataset_ns["c1_prob"].sum()

        # Choose nonschools for cluster 1 based on probability 1
        cluster1_ns_indices = np.random.choice(dataset_ns.index, size=len(cluster_1_rows), replace=False, p=dataset_ns["c1_prob"])
        cluster1_ns = dataset_ns.loc[cluster1_ns_indices]

        # Remove sampled nonschools
        dataset_ns = dataset_ns.drop(cluster1_ns_indices)

        ### CENTROID 2
        # Calculate probability to belong to centroid 2 (bivariate Gaussian)
        dataset_ns["c2_prob"] = dataset_ns["geometry"].apply(lambda row: calculate_bivar_gaussian_pdf(row, centroid2, cov_matrix2))
        dataset_ns["c2_prob"] = dataset_ns["c2_prob"] / dataset_ns["c2_prob"].sum()

        # Choose nonschools for cluster 2 based on probability 2
        cluster2_ns_indices = np.random.choice(dataset_ns.index, size=len(cluster_2_rows), replace=False, p=dataset_ns["c2_prob"])
        cluster2_ns = dataset_ns.loc[cluster2_ns_indices]

    return cluster1_ns, cluster2_ns

def plot_pr(preds_path, save_path):
    plt.figure(figsize=(8,6))
    plt.xlabel('Recall')
    plt.ylabel('Precision')

    file_name = 'train_preds.csv'
    f = os.path.join(preds_path, file_name)
    results = pd.read_csv(f)

    # Set positive class probs
    results["y_probs_pos"] = results.apply(lambda row: row["y_probs"] if row["y_preds"] == 1 else 1 - row["y_probs"], axis=1)

    precision, recall, _ = precision_recall_curve(results["y_true"], results["y_probs_pos"], pos_label=1)
    auc_score = auc(recall, precision)

    plt.plot(recall, precision)
    plt.title(f'Precision-recall curve, train (AUC = {auc_score:.2f})')
    plt.savefig(os.path.join(save_path, "PR_train.png"))

def plot_worst(cwd, anditi_dir, num_images=10):
    save_dir = os.path.join(cwd, 'worst_examples')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    file_name = 'train_preds.csv'
    f = os.path.join(cwd, file_name)
    results = pd.read_csv(f)

    # Set positive class probs
    results["y_probs_pos"] = results.apply(lambda row: row["y_probs"] if row["y_preds"] == 1 else 1 - row["y_probs"], axis=1)

    ### Non-schools
    # Filter so that only non-schools are in dataframe
    results_ns = results[results["y_true"] == 0]

    # Sort by positive class probs and take first N rows
    ns_worst_rows = results_ns.sort_values(by="y_probs_pos", ascending=False).head(num_images).reset_index(drop=True)
    ns_worst_rows.to_csv(os.path.join(save_dir, f"{num_images}_worst_ns_train.csv"))

    plt.figure()
    plt.suptitle(f'Worst non-schools, train')
    for j, row in ns_worst_rows.iterrows():
        path = f"/mnt/sdb/agorup/school_mapping/satellite_images/large/VNM/non_school/{row['UID']}.jpeg"
        image = Image.open(path).convert("RGB")

        plt.subplot(2, int(num_images / 2), j + 1)
        plt.imshow(image)
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{num_images}_worst_ns_train.png"))

    ### Schools
    # Filter so that only schools are in dataframe
    results_s = results[results["y_true"] == 1]

    # Sort by positive class probs and take first N rows
    s_worst_rows = results_s.sort_values(by="y_probs_pos", ascending=True).head(num_images).reset_index(drop=True)
    s_worst_rows.to_csv(os.path.join(save_dir, f"{num_images}_worst_s_train.csv"))

    plt.figure()
    plt.suptitle(f'Worst schools, train')
    for j, row in s_worst_rows.iterrows():
        path = f"{anditi_dir}/{row['UID']}.jpeg"
        image = Image.open(path).convert("RGB")

        plt.subplot(2, int(num_images / 2), j + 1)
        plt.imshow(image)
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{num_images}_worst_s_train.png"))

def plot_f1(cwd):
    f1_model_path = os.path.join(cwd, 'f1_model.csv')
    f1_model = pd.read_csv(f1_model_path)

    plt.figure(figsize=(8,6))
    plt.title('Train F1 over epochs')

    plt.xlabel('Epoch')
    plt.ylabel('F1 (%)')
    plt.xticks(f1_model['epoch'])

    plt.plot(f1_model['epoch'], f1_model['train'])
    plt.savefig(os.path.join(cwd, 'F1_train.png'))

def main(c, exp_name="all", sampling="inverse"):
    cwd = os.path.dirname(os.getcwd())
    exp_dir = os.path.join(cwd, c["exp_dir"], exp_name)
    print(exp_dir)
    model_file = os.path.join(exp_dir, f"model.pth")
    if not os.path.exists(model_file):
        model_file = os.path.join(exp_dir, f"{exp_name}.pth")

    finetune_dir = os.path.join(cwd, c["exp_dir"], "fine_tune_anditi", sampling)
    if not os.path.exists(finetune_dir):
        os.makedirs(finetune_dir)

    data_dir = os.path.join(finetune_dir, f"data")
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

    images_school = []
    for img in [*cluster_1, *cluster_2]:
        image_file = f"{dest_dir}/{img}.jpeg"
        images_school.append(image_file)

    phases = ["train", "test"]
    data, data_loader, classes = cnn_utils.load_dataset(config=c, phases=phases, name = "vietnam")
    data = data["train"].dataset
    data = data[data['class']=="non_school"]
    data = data[data['clean']==0]
    data = data.to_crs('EPSG:3857')

    data_ns_c1, data_ns_c2 = sample_non_schools(cluster_1_rows, cluster_2_rows, data.copy(), sampling)
    data_ns_c1.to_csv(os.path.join(data_dir, "non_schools_cluster_1.csv"))
    data_ns_c2.to_csv(os.path.join(data_dir, "non_schools_cluster_2.csv"))
    data_ns = pd.concat([data_ns_c1, data_ns_c2])

    images_non_school = []
    for i, row in data_ns.iterrows():
        image_file = f"/mnt/sdb/agorup/school_mapping/satellite_images/large/VNM/non_school/{row['UID']}.jpeg"
        images_non_school.append(image_file)

    df = pd.DataFrame(columns=["filepath","UID","class"])
    for i in range(len(images_school)):
        img = images_school[i]
        uid = img.split("/")[-1].replace(".jpeg", "")
        row = {"filepath":img, "UID": uid, "class":"school"}
        df.loc[len(df)] = row

    for i in range(len(images_non_school)):
        img = images_non_school[i]
        uid = img.split("/")[-1].replace(".jpeg", "")
        row = {"filepath":img, "UID": uid, "class":"non_school"}
        df.loc[len(df)] = row
    df.to_csv(os.path.join(finetune_dir, "df.csv"), index=False)

    dataset = SchoolDataset(df, classes_dict, train_transform)
    data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=c["batch_size"],
            num_workers=c["n_workers"],
            shuffle=True,
            drop_last=False
        )
   
    classes = ['school', 'non_school']
    n_epochs = 20
    since = time.time()

    log_string = ""
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

    f1_model = pd.DataFrame(columns=["epoch", "train"])

    train_results, _, _ = cnn_utils.evaluate(
        data_loader, 
        classes, 
        model, 
        criterion, 
        device, 
        pos_label=1,
        wandb=wandb, 
        logging=logging
    )

    f1_model.loc[len(f1_model)] = {"epoch": 0, "train": train_results["f1_score"]}

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
        log_string += "Epoch {}/{}: train F1 = {}\n".format(epoch, n_epochs, train_results['f1_score'])

        f1_model.loc[len(f1_model)] = {"epoch": epoch, "train": train_results["f1_score"]}
        
        # Terminate if learning rate becomes too low
        learning_rate = optimizer.param_groups[0]["lr"]
        if learning_rate < 1e-10:
            break

    f1_model.to_csv(os.path.join(finetune_dir, "f1_model.csv"))

    train_results, train_cm, train_preds = cnn_utils.evaluate(
            data_loader, 
            classes, 
            model, 
            criterion, 
            device, 
            pos_label=1,
            wandb=wandb, 
            logging=logging
        )
    
    train_preds.to_csv(os.path.join(finetune_dir, "train_preds.csv"))

    torch.save(model.state_dict(), os.path.join(finetune_dir, f"fine_tune_model.pth"))
    
    results_string = ""

    results_string += f"TRAIN F1: {train_results['f1_score']}\n"
    results_string += f"TRAIN PRECISION: {train_results['precision_score']}\n"
    results_string += f"TRAIN RECALL: {train_results['recall_score']}\n"
    results_string += "\n"

    print(results_string)

    f = open(os.path.join(finetune_dir, "log.txt"), "w")
    f.write(log_string)
    f.close()

    f = open(os.path.join(finetune_dir, "results.txt"), "w")
    f.write(results_string)
    f.close()

    plot_pr(finetune_dir, finetune_dir)
    plot_worst(finetune_dir, dest_dir)
    plot_f1(finetune_dir)
    
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
    parser.add_argument("--cnn_config", help="Config file", default="convnext_small")
    parser.add_argument('-d', "--device", help="device", default="cuda:0")
    parser.add_argument('-e', "--exp_name", default="global_no_vietnam_500images_no_lowres_continuous_rotation_0-90_crop352_no_AMP_convnext_small/fine_tune_vietnam_large")
    args = parser.parse_args()

    device = torch.device(args.device)

    # Load config
    config_file = os.path.join(cwd, "configs", "cnn_configs", args.cnn_config + ".yaml")
    c = config_utils.load_config(config_file)

    main(c, args.exp_name, args.sampling)