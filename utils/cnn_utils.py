import os
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import rasterio as rio
import pandas as pd
import numpy as np

import logging

logging.basicConfig(level=logging.INFO)

import timm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset

from torchvision import models, transforms
import torchvision.transforms.functional as F
from torchvision.models import (
    ResNet18_Weights,
    ResNet34_Weights,
    ResNet50_Weights,
    Inception_V3_Weights,
    VGG16_Weights,
    EfficientNet_B0_Weights,
)
import torch.nn.functional as nnf
from torch.utils.data.sampler import Sampler

import lightning as L
from torchmetrics.classification import BinaryF1Score

import sys
sys.path.insert(0, "../utils/")
import eval_utils
import clf_utils
import data_utils
import model_utils

SEED = 42

# Add temporary fix for hash error: https://github.com/pytorch/vision/issues/7744
from torchvision.models._api import WeightsEnum
from torch.hub import load_state_dict_from_url

imagenet_mean, imagenet_std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

def get_state_dict(self, *args, **kwargs):
    # kwargs.pop("check_hash")
    return load_state_dict_from_url(self.url, *args, **kwargs)
WeightsEnum.get_state_dict = get_state_dict

transform_tencrop = transforms.Compose(
        [
            transforms.CenterCrop(500),
            transforms.FiveCrop(size=352),
        ]
    )

class LightningWrapper(L.LightningModule):
    def __init__(self, encoder, use_tencrop = False):
        super().__init__()
        self.encoder = encoder
        self.f1 = BinaryF1Score()
        self.use_tencrop = use_tencrop
    
    def training_step(self, batch, batch_idx):
        x, y = batch[0], batch[1]            

        if self.use_tencrop:
            x_tencrop = transform_tencrop(x)
            y_max = torch.zeros(y.shape, device=y.device)
            x_max = None
            for xc in x_tencrop:
                if x_max is None:
                    x_max = torch.zeros(xc.shape, device = xc.device)
                y_hat = self.encoder(xc)
                soft_outputs = nnf.softmax(y_hat, dim=1).squeeze()
                positive_prob = soft_outputs[:, 1]
                x_max[positive_prob>y_max] = xc[positive_prob>y_max]
                y_max[positive_prob>y_max] = positive_prob[positive_prob>y_max]
            x = x_max
        
        y_hat = self.encoder(x)
        loss = nnf.cross_entropy(y_hat, y, label_smoothing=0.1)

        outputs = self.encoder(x)
        soft_outputs = nnf.softmax(outputs, dim=1)
        positive_probs = soft_outputs[:, 1]
        f1 = self.f1(positive_probs, y)
        self.log_dict({"train_loss": loss, "train_F1": f1}, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch[0], batch[1]
        y_hat = self.encoder(x)
        val_loss = nnf.cross_entropy(y_hat, y)

        outputs = self.encoder(x)
        soft_outputs = nnf.softmax(outputs, dim=1)
        positive_probs = soft_outputs[:, 1]
        f1 = self.f1(positive_probs, y)
        self.log_dict({"val_loss": val_loss, "val_F1": f1}, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return {"val_loss": val_loss, "val_F1": f1}

    def test_step(self, batch, batch_idx):
        x, y = batch[0], batch[1]
        y_hat = self.encoder(x)
        test_loss = nnf.cross_entropy(y_hat, y)
        self.log("test_loss", test_loss, sync_dist=True)

    def predict_step(self, batch, batch_idx):
        x, y = batch
        pred = self.encoder(x)
        return pred
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.encoder.parameters(), lr=1e-5)
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.1, patience=7, mode='max'
        )
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler, "monitor": "val_F1"}
        return [optimizer], [lr_scheduler]

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

class WeightedSchoolDataset(Dataset):
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
        self.dataset_school = dataset[dataset["class"] == "school"]
        self.num_schools = len(self.dataset_school)
        self.dataset_non_school = dataset[dataset["class"] == "non_school"]
        self.num_non_schools = len(self.dataset_non_school)
        self.bigget_dataset_size = self.num_non_schools if self.num_non_schools > self.num_schools else self.num_schools
        self.smaller_dataset_size = self.num_schools if self.num_non_schools > self.num_schools else self.num_non_schools
        self.bigger_dataset = self.dataset_non_school if self.num_non_schools > self.num_schools else self.dataset_school
        self.smaller_dataset = self.dataset_school if self.num_non_schools > self.num_schools else self.dataset_non_school
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
        
        if index >= self.bigget_dataset_size:
            item = self.smaller_dataset.iloc[(index - self.bigget_dataset_size) % self.smaller_dataset_size]
        else:
            item = self.bigger_dataset.iloc[index]

        #item = self.dataset.iloc[index]
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
        
        #return len(self.dataset)
        return 2 * self.bigget_dataset_size


def visualize_data(data, data_loader, phase="test", n=4):
    """
    Visualize a sample of data from a DataLoader.

    Args:
    - data (dict): A dictionary containing data split into different phases 
    (TRAIN, VALIDATION, TEST).
    - data_loader (torch.utils.data.DataLoader): DataLoader containing the data.
    - phase (str, optional): The phase of data to visualize. Defaults to "test".
    - n (int, optional): Number of images to visualize in a grid. Defaults to 4.
    """
    
    inputs, classes, uids = next(iter(data_loader[phase]))
    fig, axes = plt.subplots(n, n, figsize=(6, 6))

    key_list = list(data[phase].classes.keys())
    val_list = list(data[phase].classes.values())

    for i in range(n):
        for j in range(n):
            image = inputs[i * n + j].numpy().transpose((1, 2, 0))
            title = key_list[val_list.index(classes[i * n + j])]
            image = np.clip(
                np.array(imagenet_std) * image + np.array(imagenet_mean), 0, 1
            )
            axes[i, j].imshow(image)
            axes[i, j].set_title(title, fontdict={"fontsize": 7})
            axes[i, j].axis("off")


def remove_data_without_images(dataset):
    mask = dataset["filepath"].apply(lambda filepath: os.path.exists(filepath) and os.path.getsize(filepath) > 10000)
    filtered_dataset = dataset[mask]
    return filtered_dataset

def load_dataset(config, phases, name=None):
    """
    Load dataset based on configuration settings and phases.

    Args:
    - config (dict): Configuration settings including data directories, attributes, etc.
    - phases (list): List of phases for which to load the dataset (e.g., ["train", "test"]).
    - prefix (str, optional): Prefix to be added to file paths. Defaults to an empty string.

    Returns:
    - tuple: A tuple containing:
        - dict: A dictionary containing datasets for each phase.
        - dict: A dictionary containing data loaders for each phase.
        - dict: A dictionary containing classes and their mappings.
    """
    
    dataset = model_utils.load_data(config, attributes=["rurban", "iso"], verbose=False, name=name)
    dataset["filepath"] = data_utils.get_image_filepaths(config, dataset)

    # Remove data entries which do not have a corresponding sattelite image file
    dataset = remove_data_without_images(dataset)

    classes_dict = {config["pos_class"] : 1, config["neg_class"]: 0}

    transforms = get_transforms(size=config["img_size"])
    classes = list(dataset["class"].unique())
    logging.info(f" Classes: {classes}")

    #sampler = SchoolSampler(dataset, 14)
    # data = {
    #     phase: WeightedSchoolDataset(
    #         dataset[dataset.dataset==phase]
    #         .sample(frac=1, random_state=SEED)
    #         .reset_index(drop=True),
    #         classes_dict,
    #         transforms[phase]
    #     ) if phase == "train" else SchoolDataset(
    #         dataset[dataset.dataset==phase]
    #         .sample(frac=1, random_state=SEED)
    #         .reset_index(drop=True),
    #         classes_dict,
    #         transforms[phase]
    #     )
    #     for phase in phases
    # }

    data = {
        phase: SchoolDataset(
            dataset[dataset.dataset==phase]
            .sample(frac=1, random_state=SEED)
            .reset_index(drop=True),
            classes_dict,
            transforms[phase]
        ) 
        for phase in phases
    }

    data_loader = {
        phase: torch.utils.data.DataLoader(
            data[phase],
            batch_size=config["batch_size"],
            num_workers=config["n_workers"],
            shuffle=True if phase == "train" else False,
            drop_last=True if phase == "train" else False
        )
        for phase in phases
    }

    return data, data_loader, classes


def train_tencrop(data_loader, model, criterion, optimizer, device, logging, pos_label, wandb=None):
    """
    Train the model on the provided data.

    Args:
    - data_loader (torch.utils.data.DataLoader): DataLoader containing training data.
    - model (torch.nn.Module): The neural network model.
    - criterion: Loss function.
    - optimizer: Optimization algorithm.
    - device (str): Device to run the training on (e.g., 'cuda' or 'cpu').
    - logging: Logging object to record training information.
    - wandb: Weights & Biases object for logging if available. Defaults to None.

    Returns:
    - dict: Results of the training including loss and evaluation metrics.
    """
    
    model.train()

    y_actuals, y_preds = [], []
    running_loss = 0.0
    for inputs, labels, _ in tqdm(data_loader, total=len(data_loader)):
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        with torch.set_grad_enabled(True):
            #with torch.autocast(device_type="cuda", dtype=torch.float16):
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            y_actuals.extend(labels.cpu().numpy().tolist())
            y_preds.extend(preds.data.cpu().numpy().tolist())

    epoch_loss = running_loss / len(data_loader)
    epoch_results = eval_utils.evaluate(y_actuals, y_preds, pos_label)
    epoch_results["loss"] = epoch_loss

    learning_rate = optimizer.param_groups[0]["lr"]
    logging.info(f"Train Loss: {epoch_loss} {epoch_results} LR: {learning_rate}")

    #if wandb is not None:
        #wandb.log({"train_" + k: v for k, v in epoch_results.items()})
    return epoch_results

def train(data_loader, model, criterion, optimizer, device, logging, pos_label, wandb=None):
    """
    Train the model on the provided data.

    Args:
    - data_loader (torch.utils.data.DataLoader): DataLoader containing training data.
    - model (torch.nn.Module): The neural network model.
    - criterion: Loss function.
    - optimizer: Optimization algorithm.
    - device (str): Device to run the training on (e.g., 'cuda' or 'cpu').
    - logging: Logging object to record training information.
    - wandb: Weights & Biases object for logging if available. Defaults to None.

    Returns:
    - dict: Results of the training including loss and evaluation metrics.
    """
    
    model.train()

    y_actuals, y_preds = [], []
    running_loss = 0.0
    for inputs, labels, _ in tqdm(data_loader, total=len(data_loader)):
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        with torch.set_grad_enabled(True):
            #with torch.autocast(device_type="cuda", dtype=torch.float16):
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            y_actuals.extend(labels.cpu().numpy().tolist())
            y_preds.extend(preds.data.cpu().numpy().tolist())

    epoch_loss = running_loss / len(data_loader)
    epoch_results = eval_utils.evaluate(y_actuals, y_preds, pos_label)
    epoch_results["loss"] = epoch_loss

    learning_rate = optimizer.param_groups[0]["lr"]
    logging.info(f"Train Loss: {epoch_loss} {epoch_results} LR: {learning_rate}")

    #if wandb is not None:
        #wandb.log({"train_" + k: v for k, v in epoch_results.items()})
    return epoch_results


def evaluate(data_loader, class_names, model, criterion, device, logging, pos_label, wandb=None, threshold=0.5):
    """
    Evaluate the model using the provided data.

    Args:
    - data_loader (torch.utils.data.DataLoader): DataLoader containing validation/test data.
    - class_names (list): List of class names.
    - model (torch.nn.Module): The neural network model.
    - criterion: Loss function.
    - device (str): Device to run evaluation on (e.g., 'cuda' or 'cpu').
    - logging: Logging object to record evaluation information.
    - wandb: Weights & Biases object for logging if available. Defaults to None.

    Returns:
    - tuple: A tuple containing:
        - dict: Results of the evaluation including loss and evaluation metrics.
        - tuple: A tuple containing confusion matrix, metrics, and report.
    """
    
    model.eval()

    y_uids, y_actuals, y_preds, y_probs = [], [], [], []
    running_loss = 0.0
    confusion_matrix = torch.zeros(len(class_names), len(class_names))

    for inputs, labels, uids in tqdm(data_loader, total=len(data_loader)):
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.set_grad_enabled(False):
            #with torch.autocast(device_type="cuda", dtype=torch.float16):
            outputs = model(inputs)
            soft_outputs = nnf.softmax(outputs, dim=1)
            positive_probs = soft_outputs[:, 1]
            probs, _ = soft_outputs.topk(1, dim=1)
            #_, preds = torch.max(outputs, 1)
            preds = (positive_probs > threshold).int()
            
            loss = criterion(outputs, labels)

        running_loss += loss.item() * inputs.size(0)
        y_actuals.extend(labels.cpu().numpy().tolist())
        y_preds.extend(preds.data.cpu().numpy().tolist())
        y_probs.extend(probs.data.cpu().numpy().tolist())
        y_uids.extend(uids)

    epoch_loss = running_loss / len(data_loader)
    epoch_results = eval_utils.evaluate(y_actuals, y_preds, pos_label)
    epoch_results["loss"] = epoch_loss

    confusion_matrix, cm_metrics, cm_report = eval_utils.get_confusion_matrix(
        y_actuals, y_preds, class_names
    )
    y_probs = [x[0] for x in y_probs]
    logging.info(f"Val Loss: {epoch_loss} {epoch_results}")
    preds = pd.DataFrame({
        'UID': y_uids,
        'y_true': y_actuals, 
        'y_preds': y_preds, 
        'y_probs': y_probs
    })

    #if wandb is not None:
        #wandb.log({"val_" + k: v for k, v in epoch_results.items()})
    return epoch_results, (confusion_matrix, cm_metrics, cm_report), preds


def get_transforms(size):
    """
    Get image transformations for training and testing phases.

    Args:
    - size (int): Size of the transformed images.

    Returns:
    - dict: A dictionary containing transformation pipelines for "TRAIN" and "TEST" phases.
    """

    return {
        "train": transforms.Compose(
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
        ),
        "test": transforms.Compose(
            [
                
                #transforms.Resize(size),
                #transforms.CenterCrop(size),
                transforms.CenterCrop(500),
                transforms.ToTensor(),
                transforms.Normalize(imagenet_mean, imagenet_std),
            ]
        ),
    }


def get_model(model_type, n_classes, dropout=0):
    """
    Get a neural network model based on specified parameters.

    Args:
    - model_type (str): The type of model architecture to use.
    - n_classes (int): The number of output classes.
    - dropout (float, optional): Dropout rate if applicable. Defaults to 0.

    Returns:
    - torch.nn.Module: A neural network model based on the specified architecture.
    """
    
    if "resnet" in model_type:
        if model_type == "resnet18":
            model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        elif model_type == "resnet34":
            model = models.resnet34(weights=ResNet34_Weights.DEFAULT)
        elif model_type == "resnet50":
            model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        num_ftrs = model.fc.in_features
        if dropout > 0:
            model.fc = nn.Sequential(
                nn.Dropout(dropout), nn.Linear(num_ftrs, n_classes)
            )
        else:
            model.fc = nn.Linear(num_ftrs, n_classes)

    if "inception" in model_type:
        model = models.inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)
        model.aux_logits = False
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, n_classes)

    if "vgg" in model_type:
        model = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs, n_classes)

    if "efficientnet" in model_type:
        model = models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, n_classes)

    if "xception" in model_type:
        model = timm.create_model('xception', pretrained=True, num_classes=n_classes)

    if "convnext" in model_type:
        if "small" in model_type:
            model = models.convnext_small(weights='IMAGENET1K_V1')
        elif "base" in model_type:
            model = models.convnext_base(weights='IMAGENET1K_V1')
        elif "large" in model_type:
            model = models.convnext_large(weights='IMAGENET1K_V1')
        num_ftrs = model.classifier[2].in_features
        model.classifier[2] = nn.Linear(num_ftrs, n_classes)
    
    return model


def load_model(
    model_type,
    n_classes,
    pretrained,
    scheduler_type,
    optimizer_type,
    label_smoothing=0.0,
    lr=0.001,
    momentum=0.9,
    gamma=0.1,
    step_size=7,
    patience=7,
    dropout=0,
    device="cpu",
):
    """
    Load a neural network model with specified configurations.

    Args:
    - model_type (str): The type of model architecture to use.
    - n_classes (int): The number of output classes.
    - pretrained (bool): Whether to use pre-trained weights.
    - scheduler_type (str): The type of learning rate scheduler to use.
    - optimizer_type (str): The type of optimizer to use.
    - label_smoothing (float, optional): Label smoothing parameter. Defaults to 0.0.
    - lr (float, optional): Learning rate. Defaults to 0.001.
    - momentum (float, optional): Momentum factor for SGD optimizer. Defaults to 0.9.
    - gamma (float, optional): Gamma factor for learning rate scheduler. Defaults to 0.1.
    - step_size (int, optional): Step size for learning rate scheduler. Defaults to 7.
    - patience (int, optional): Patience for ReduceLROnPlateau scheduler. Defaults to 7.
    - dropout (float, optional): Dropout rate if applicable. Defaults to 0.
    - device (str, optional): Device to run the model on. Defaults to "cpu".

    Returns:
    - tuple: A tuple containing the loaded model, criterion, optimizer, and scheduler.
    """
    
    model = get_model(model_type, n_classes, dropout)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    if optimizer_type == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    elif optimizer_type == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    if scheduler_type == "StepLR":
        scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif scheduler_type == "ReduceLROnPlateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.1, patience=patience, mode='max'
        )

    return model, criterion, optimizer, scheduler

def load_model_lightning(
    model_type,
    n_classes,
    pretrained,
    scheduler_type,
    optimizer_type,
    label_smoothing=0.0,
    lr=0.001,
    momentum=0.9,
    gamma=0.1,
    step_size=7,
    patience=7,
    dropout=0,
    device="cpu",
    use_tencrop = False
):
    """
    Load a neural network model with specified configurations.

    Args:
    - model_type (str): The type of model architecture to use.
    - n_classes (int): The number of output classes.
    - pretrained (bool): Whether to use pre-trained weights.
    - scheduler_type (str): The type of learning rate scheduler to use.
    - optimizer_type (str): The type of optimizer to use.
    - label_smoothing (float, optional): Label smoothing parameter. Defaults to 0.0.
    - lr (float, optional): Learning rate. Defaults to 0.001.
    - momentum (float, optional): Momentum factor for SGD optimizer. Defaults to 0.9.
    - gamma (float, optional): Gamma factor for learning rate scheduler. Defaults to 0.1.
    - step_size (int, optional): Step size for learning rate scheduler. Defaults to 7.
    - patience (int, optional): Patience for ReduceLROnPlateau scheduler. Defaults to 7.
    - dropout (float, optional): Dropout rate if applicable. Defaults to 0.
    - device (str, optional): Device to run the model on. Defaults to "cpu".

    Returns:
    - tuple: A tuple containing the loaded model, criterion, optimizer, and scheduler.
    """
    
    model = get_model(model_type, n_classes, dropout)
    model = LightningWrapper(model, use_tencrop=use_tencrop)
    return model