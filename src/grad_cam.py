from PIL import Image
from torchvision import models, transforms

import sys
sys.path.insert(0, "../utils/")
import cnn_utils
import config_utils
import os

import torchvision.transforms as transforms
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import numpy as np
import pandas
import torch.nn.functional as nnf

imagenet_mean, imagenet_std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
size = 224

if __name__ == "__main__":
    device = "cuda:0"
    images_dir = "/mnt/ssd1/agorup/school_mapping/satellite_images"

    cwd = os.path.dirname(os.getcwd())
    config_file = os.path.join(cwd, "configs", "cnn_configs", "resnet18.yaml")
    exp_name = "all_no_resize_resnet18"
    exp_dir = os.path.join(cwd, "exp", exp_name)
    c = config_utils.load_config(config_file)

    csv_file = os.path.join(exp_dir, exp_name + ".csv")
    csv_file = os.path.join(exp_dir, "THA", "THA.csv")
    csv = pandas.read_csv(csv_file)

    school_correct = []
    school_incorrect = []
    nonschool_correct = []
    nonschool_incorrect = []
    for i in csv.index:
        example = csv["UID"][i]
        
        if csv["y_true"][i] != csv["y_preds"][i]:
            if "NON_SCHOOL" in example: 
                nonschool_incorrect.append(i)
            else:
                school_incorrect.append(i)

        else:
            if "NON_SCHOOL" in example: 
                nonschool_correct.append(i)
            else:
                school_correct.append(i)

    school_correct_probs = csv["y_probs"][school_correct]
    school_incorrect_probs = csv["y_probs"][school_incorrect]
    nonschool_correct_probs = csv["y_probs"][nonschool_correct]
    nonschool_incorrect_probs = csv["y_probs"][nonschool_incorrect]

    school_correct_argsort = np.argsort(school_correct_probs)
    school_incorrect_argsort = np.argsort(school_incorrect_probs)
    nonschool_correct_argsort = np.argsort(nonschool_correct_probs)
    nonschool_incorrect_argsort = np.argsort(nonschool_incorrect_probs)

    best_school_correct = np.array(school_correct)[school_correct_argsort[-10:]]
    best_school_incorrect = np.array(school_incorrect)[school_incorrect_argsort[-5:]]
    best_nonschool_correct = np.array(nonschool_correct)[nonschool_correct_argsort[-10:]]
    best_nonschool_incorrect = np.array(nonschool_incorrect)[nonschool_incorrect_argsort[-5:]]

    random_school_correct = np.random.choice(school_correct, 5, replace=False)
    random_school_incorrect = np.random.choice(school_incorrect, 5, replace=False)
    random_nonschool_correct = np.random.choice(nonschool_correct, 5, replace=False)
    random_nonschool_incorrect = np.random.choice(nonschool_incorrect, 5, replace=False)


    model, criterion, optimizer, scheduler = cnn_utils.load_model(
        n_classes=2,
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

    model_file = os.path.join(exp_dir, exp_name + ".pth")
    model.load_state_dict(torch.load(model_file, map_location=device))
    model = model.to(device)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean, imagenet_std),
    ])

    transform2 = transforms.Compose([
        transforms.ToTensor(),
    ])

    exp_dir = os.path.join(cwd, "exp", exp_name, "THA")
    if not os.path.exists(os.path.join(exp_dir, "gradcam")):
            os.makedirs(os.path.join(exp_dir, "gradcam"))
    
    if not os.path.exists(os.path.join(exp_dir, "gradcam", "images")):
            os.makedirs(os.path.join(exp_dir, "gradcam", "images"))

    target_layers = [model.layer4[-1]]
    cam = GradCAM(model=model, target_layers=target_layers)

    cot = 1
    targets = [ClassifierOutputTarget(cot)]
    
    for i in best_school_correct:
        UID = csv["UID"][i]
        ISO = UID[4:7]
        image_path = os.path.join(images_dir, ISO, "school", UID + ".jpeg")
        image = Image.open(image_path).convert("RGB")
        image.save(os.path.join(exp_dir, "gradcam", "images", f"best_school_{UID}.jpeg"))
        img = transform(image)
        img = img.unsqueeze(0).to(device)
        out = model(img)
        img = img.to(device)
        grayscale_cam = cam(input_tensor=img, targets=targets)
        grayscale_cam = grayscale_cam[0, :]
        img2 = transform2(image)
        rgb_img = img2.cpu().numpy().transpose(1, 2, 0)
        visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
        image = Image.fromarray(visualization)
        image.save(os.path.join(exp_dir, "gradcam", f"gradcam_best_school_COT{cot}_{UID}.jpeg"))
    for i in best_nonschool_incorrect:
        UID = csv["UID"][i]
        ISO = UID[4:7]
        image_path = os.path.join(images_dir, ISO, "non_school", UID + ".jpeg")
        image = Image.open(image_path).convert("RGB")
        image.save(os.path.join(exp_dir, "gradcam", "images", f"worst_nonschool_{UID}.jpeg"))
        img = transform(image)
        img = img.unsqueeze(0)
        img = img.to(device)
        grayscale_cam = cam(input_tensor=img, targets=targets)
        grayscale_cam = grayscale_cam[0, :]
        img2 = transform2(image)
        rgb_img = img2.cpu().numpy().transpose(1, 2, 0)
        visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
        image = Image.fromarray(visualization)
        image.save(os.path.join(exp_dir, "gradcam", f"gradcam_worst_nonschool_COT{cot}_{UID}.jpeg"))
    for i in best_school_incorrect:
        UID = csv["UID"][i]
        ISO = UID[4:7]
        image_path = os.path.join(images_dir, ISO, "school", UID + ".jpeg")
        image = Image.open(image_path).convert("RGB")
        image.save(os.path.join(exp_dir, "gradcam", "images", f"worst_school_{UID}.jpeg"))
        img = transform(image)
        img = img.unsqueeze(0)
        img = img.to(device)
        grayscale_cam = cam(input_tensor=img, targets=targets)
        grayscale_cam = grayscale_cam[0, :]
        img2 = transform2(image)
        rgb_img = img2.cpu().numpy().transpose(1, 2, 0)
        visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
        image = Image.fromarray(visualization)
        image.save(os.path.join(exp_dir, "gradcam", f"gradcam_worst_school_COT{cot}_{UID}.jpeg"))
    for i in best_nonschool_correct:
        UID = csv["UID"][i]
        ISO = UID[4:7]
        image_path = os.path.join(images_dir, ISO, "non_school", UID + ".jpeg")
        image = Image.open(image_path).convert("RGB")
        image.save(os.path.join(exp_dir, "gradcam", "images", f"best_nonschool_{UID}.jpeg"))
        img = transform(image)
        img = img.unsqueeze(0)
        img = img.to(device)
        grayscale_cam = cam(input_tensor=img, targets=targets)
        grayscale_cam = grayscale_cam[0, :]
        img2 = transform2(image)
        rgb_img = img2.cpu().numpy().transpose(1, 2, 0)
        visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
        image = Image.fromarray(visualization)
        image.save(os.path.join(exp_dir, "gradcam", f"gradcam_best_nonschool_COT{cot}_{UID}.jpeg"))
    # RANDOM IMAGES
    for i in random_school_correct:
        UID = csv["UID"][i]
        ISO = UID[4:7]
        image_path = os.path.join(images_dir, ISO, "school", UID + ".jpeg")
        image = Image.open(image_path).convert("RGB")
        image.save(os.path.join(exp_dir, "gradcam", "images", f"random_good_school_{UID}.jpeg"))
        img = transform(image)
        img = img.unsqueeze(0)
        img = img.to(device)
        grayscale_cam = cam(input_tensor=img, targets=targets)
        grayscale_cam = grayscale_cam[0, :]
        img2 = transform2(image)
        rgb_img = img2.cpu().numpy().transpose(1, 2, 0)
        visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
        image = Image.fromarray(visualization)
        image.save(os.path.join(exp_dir, "gradcam", f"gradcam_random_good_school_COT{cot}_{UID}.jpeg"))
    for i in random_school_incorrect:
        UID = csv["UID"][i]
        ISO = UID[4:7]
        image_path = os.path.join(images_dir, ISO, "school", UID + ".jpeg")
        image = Image.open(image_path).convert("RGB")
        image.save(os.path.join(exp_dir, "gradcam", "images", f"random_bad_school_{UID}.jpeg"))
        img = transform(image)
        img = img.unsqueeze(0)
        img = img.to(device)
        grayscale_cam = cam(input_tensor=img, targets=targets)
        grayscale_cam = grayscale_cam[0, :]
        img2 = transform2(image)
        rgb_img = img2.cpu().numpy().transpose(1, 2, 0)
        visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
        image = Image.fromarray(visualization)
        image.save(os.path.join(exp_dir, "gradcam", f"gradcam_random_bad_school_COT{cot}_{UID}.jpeg"))
    for i in random_nonschool_correct:
        UID = csv["UID"][i]
        ISO = UID[4:7]
        image_path = os.path.join(images_dir, ISO, "non_school", UID + ".jpeg")
        image = Image.open(image_path).convert("RGB")
        image.save(os.path.join(exp_dir, "gradcam", "images", f"random_good_nonschool_{UID}.jpeg"))
        img = transform(image)
        img = img.unsqueeze(0)
        img = img.to(device)
        grayscale_cam = cam(input_tensor=img, targets=targets)
        grayscale_cam = grayscale_cam[0, :]
        img2 = transform2(image)
        rgb_img = img2.cpu().numpy().transpose(1, 2, 0)
        visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
        image = Image.fromarray(visualization)
        image.save(os.path.join(exp_dir, "gradcam", f"gradcam_random_good_nonschool_COT{cot}_{UID}.jpeg"))
    for i in random_nonschool_correct:
        UID = csv["UID"][i]
        ISO = UID[4:7]
        image_path = os.path.join(images_dir, ISO, "non_school", UID + ".jpeg")
        image = Image.open(image_path).convert("RGB")
        image.save(os.path.join(exp_dir, "gradcam", "images", f"random_bad_nonschool_{UID}.jpeg"))
        img = transform(image)
        img = img.unsqueeze(0)
        img = img.to(device)
        grayscale_cam = cam(input_tensor=img, targets=targets)
        grayscale_cam = grayscale_cam[0, :]
        img2 = transform2(image)
        rgb_img = img2.cpu().numpy().transpose(1, 2, 0)
        visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
        image = Image.fromarray(visualization)
        image.save(os.path.join(exp_dir, "gradcam", f"gradcam_random_bad_nonschool_COT{cot}_{UID}.jpeg"))




    cot = 0
    targets = [ClassifierOutputTarget(cot)]
    for i in best_school_correct:
        UID = csv["UID"][i]
        ISO = UID[4:7]
        image_path = os.path.join(images_dir, ISO, "school", UID + ".jpeg")
        image = Image.open(image_path).convert("RGB")
        img = transform(image)
        img = img.unsqueeze(0)
        img = img.to(device)
        grayscale_cam = cam(input_tensor=img, targets=targets)
        grayscale_cam = grayscale_cam[0, :]
        img2 = transform2(image)
        rgb_img = img2.cpu().numpy().transpose(1, 2, 0)
        visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
        image = Image.fromarray(visualization)
        image.save(os.path.join(exp_dir, "gradcam", f"gradcam_best_school_COT{cot}_{UID}.jpeg"))
    for i in best_nonschool_incorrect:
        UID = csv["UID"][i]
        ISO = UID[4:7]
        image_path = os.path.join(images_dir, ISO, "non_school", UID + ".jpeg")
        image = Image.open(image_path).convert("RGB")
        img = transform(image)
        img = img.unsqueeze(0)
        img = img.to(device)
        grayscale_cam = cam(input_tensor=img, targets=targets)
        grayscale_cam = grayscale_cam[0, :]
        img2 = transform2(image)
        rgb_img = img2.cpu().numpy().transpose(1, 2, 0)
        visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
        image = Image.fromarray(visualization)
        image.save(os.path.join(exp_dir, "gradcam", f"gradcam_worst_nonschool_COT{cot}_{UID}.jpeg"))
    for i in best_school_incorrect:
        UID = csv["UID"][i]
        ISO = UID[4:7]
        image_path = os.path.join(images_dir, ISO, "school", UID + ".jpeg")
        image = Image.open(image_path).convert("RGB")
        img = transform(image)
        img = img.unsqueeze(0)
        img = img.to(device)
        grayscale_cam = cam(input_tensor=img, targets=targets)
        grayscale_cam = grayscale_cam[0, :]
        img2 = transform2(image)
        rgb_img = img2.cpu().numpy().transpose(1, 2, 0)
        visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
        image = Image.fromarray(visualization)
        image.save(os.path.join(exp_dir, "gradcam", f"gradcam_worst_school_COT{cot}_{UID}.jpeg"))
    for i in best_nonschool_correct:
        UID = csv["UID"][i]
        ISO = UID[4:7]
        image_path = os.path.join(images_dir, ISO, "non_school", UID + ".jpeg")
        image = Image.open(image_path).convert("RGB")
        img = transform(image)
        img = img.unsqueeze(0)
        img = img.to(device)
        grayscale_cam = cam(input_tensor=img, targets=targets)
        grayscale_cam = grayscale_cam[0, :]
        img2 = transform2(image)
        rgb_img = img2.cpu().numpy().transpose(1, 2, 0)
        visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
        image = Image.fromarray(visualization)
        image.save(os.path.join(exp_dir, "gradcam", f"gradcam_best_nonschool_COT{cot}_{UID}.jpeg"))
    # RANDOM IMAGES
    for i in random_school_correct:
        UID = csv["UID"][i]
        ISO = UID[4:7]
        image_path = os.path.join(images_dir, ISO, "school", UID + ".jpeg")
        image = Image.open(image_path).convert("RGB")
        img = transform(image)
        img = img.unsqueeze(0)
        img = img.to(device)
        grayscale_cam = cam(input_tensor=img, targets=targets)
        grayscale_cam = grayscale_cam[0, :]
        img2 = transform2(image)
        rgb_img = img2.cpu().numpy().transpose(1, 2, 0)
        visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
        image = Image.fromarray(visualization)
        image.save(os.path.join(exp_dir, "gradcam", f"gradcam_random_good_school_COT{cot}_{UID}.jpeg"))
    for i in random_school_incorrect:
        UID = csv["UID"][i]
        ISO = UID[4:7]
        image_path = os.path.join(images_dir, ISO, "school", UID + ".jpeg")
        image = Image.open(image_path).convert("RGB")
        img = transform(image)
        img = img.unsqueeze(0)
        img = img.to(device)
        grayscale_cam = cam(input_tensor=img, targets=targets)
        grayscale_cam = grayscale_cam[0, :]
        img2 = transform2(image)
        rgb_img = img2.cpu().numpy().transpose(1, 2, 0)
        visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
        image = Image.fromarray(visualization)
        image.save(os.path.join(exp_dir, "gradcam", f"gradcam_random_bad_school_COT{cot}_{UID}.jpeg"))
    for i in random_nonschool_correct:
        UID = csv["UID"][i]
        ISO = UID[4:7]
        image_path = os.path.join(images_dir, ISO, "non_school", UID + ".jpeg")
        image = Image.open(image_path).convert("RGB")
        img = transform(image)
        img = img.unsqueeze(0)
        img = img.to(device)
        grayscale_cam = cam(input_tensor=img, targets=targets)
        grayscale_cam = grayscale_cam[0, :]
        img2 = transform2(image)
        rgb_img = img2.cpu().numpy().transpose(1, 2, 0)
        visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
        image = Image.fromarray(visualization)
        image.save(os.path.join(exp_dir, "gradcam", f"gradcam_random_good_nonschool_COT{cot}_{UID}.jpeg"))
    for i in random_nonschool_correct:
        UID = csv["UID"][i]
        ISO = UID[4:7]
        image_path = os.path.join(images_dir, ISO, "non_school", UID + ".jpeg")
        image = Image.open(image_path).convert("RGB")
        img = transform(image)
        img = img.unsqueeze(0)
        img = img.to(device)
        grayscale_cam = cam(input_tensor=img, targets=targets)
        grayscale_cam = grayscale_cam[0, :]
        img2 = transform2(image)
        rgb_img = img2.cpu().numpy().transpose(1, 2, 0)
        visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
        image = Image.fromarray(visualization)
        image.save(os.path.join(exp_dir, "gradcam", f"gradcam_random_bad_nonschool_COT{cot}_{UID}.jpeg"))

