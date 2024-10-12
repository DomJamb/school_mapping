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

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
import warnings
warnings.filterwarnings("ignore", message="CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR")

# Get device
cwd = os.path.dirname(os.getcwd())
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
torch.set_float32_matmul_precision('high')
SEED = 42
checkpoint_callback = ModelCheckpoint(
        monitor='val_F1',
        mode='max',  # maximize the F1 score
        save_top_k=1,  # save only the best model
        filename='model',
        verbose=True,
        save_weights_only=True
    )
trainer = L.Trainer(accelerator="gpu", devices=[2], max_epochs=20, precision="bf16", callbacks=[checkpoint_callback], strategy="ddp")

def main(c, exp_name="all", dataset_name=None):    
    # Create experiment folder
    #exp_name = f"{c['iso_code']}_{c['config_name']}"
    exp_name = f"{dataset_name}_{exp_name}_{c['config_name']}_lightning" if dataset_name is not None else f"{exp_name}_{c['config_name']}_lightning"
    exp_dir = os.path.join(cwd, c["exp_dir"], exp_name)
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
    
    log_string = ""
    logname = os.path.join(exp_dir, f"{exp_name}.log")
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    handler = logging.FileHandler(logname)
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)
    

    # Set wandb configs
    #wandb.init(project="UNICEFv2", config=c)
    #wandb.run.name = exp_name
    #wandb.config = c
    
    # Load dataset
    phases = ["train", "test"]
    data, data_loader, classes = cnn_utils.load_dataset(config=c, phases=phases, name = dataset_name)
    if trainer.global_rank == 0:
        logging.info(exp_name)
        logging.info(f"Train/test sizes: {len(data['train'])}/{len(data['test'])}")

    # Load model, optimizer, and scheduler
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

    model = cnn_utils.load_model_lightning(
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
        use_tencrop=False
    )

    
    checkpoint_callback.dirpath = exp_dir
    since = time.time()
    
    trainer.fit(model, data_loader["train"], data_loader["test"])


    if trainer.global_rank == 0:
        model_file = os.path.join(exp_dir, "model.ckpt")
        new_model_file = os.path.join(exp_dir, "model.pth")
        if(os.path.exists(model_file)):
            os.rename(model_file, new_model_file)

    

        # Commence model training
        #n_epochs = c["n_epochs"]
        
        #best_score = -1

        # for epoch in range(1, n_epochs + 1):
        #     logging.info("\nEpoch {}/{}".format(epoch, n_epochs))

        #     # Train model
        #     train_results = cnn_utils.train(
        #         data_loader["train"],
        #         model,
        #         criterion,
        #         optimizer,
        #         device,
        #         pos_label=1,
        #         wandb=wandb,
        #         logging=logging
        #     )

        #     log_string += f"epoch {epoch}:  train F1 = {train_results['f1_score']}, "

        #     # Evauate model
        #     val_results, val_cm, val_preds = cnn_utils.evaluate(
        #         data_loader["test"], 
        #         classes, 
        #         model, 
        #         criterion, 
        #         device, 
        #         pos_label=1,
        #         wandb=wandb, 
        #         logging=logging
        #     )
        #     scheduler.step(val_results["f1_score"])

        #     log_string += f"test F1 = {val_results['f1_score']}\n"

        #     # Save best model so far
        #     if val_results["f1_score"] > best_score:
        #         best_score = val_results["f1_score"]
        #         best_weights = model.state_dict()

        #         eval_utils._save_files(val_results, val_cm, exp_dir)
        #         model_file = os.path.join(exp_dir, f"{exp_name}.pth")
        #         torch.save(model.state_dict(), model_file)
        #     logging.info(f"Best F1 score: {best_score}")

        #     # Terminate if learning rate becomes too low
        #     learning_rate = optimizer.param_groups[0]["lr"]
        #     if learning_rate < 1e-10:
        #         break

        # Terminate trackers
        time_elapsed = time.time() - since
        logging.info(
            "Training complete in {:.0f}m {:.0f}s".format(
                time_elapsed // 60, time_elapsed % 60
            )
        )

        # log_string += "Training complete in {:.0f}m {:.0f}s\n".format(
        #         time_elapsed // 60, time_elapsed % 60
        #     )
        # log_file = os.path.join(exp_dir, "log.txt")
        # f = open(log_file, "w")
        # f.write(log_string)
        # f.close()

        # Load best model
        model_file = os.path.join(exp_dir, f"model.pth")
        # model1.load_state_dict(torch.load(model_file, map_location=device))
        # model = model1.to(device)

        model = cnn_utils.LightningWrapper.load_from_checkpoint(model_file, encoder=model.encoder).encoder.to(device)

        # Calculate test performance using best model
        logging.info("\nTest Results")
        test_results, test_cm, test_preds = cnn_utils.evaluate(
            data_loader["test"], classes, model, criterion, device, pos_label=1, wandb=wandb, logging=logging
        )
        test_preds.to_csv(os.path.join(exp_dir, f"{exp_name}.csv"), index=False)

        # Save results in experiment directory
        eval_utils._save_files(test_results, test_cm, exp_dir)



if __name__ == "__main__":
    # Parser
    parser = argparse.ArgumentParser(description="Model Training")
    parser.add_argument('-c', "--cnn_config", help="Config file", default="resnet18")
    parser.add_argument("--iso", help="ISO code", default=[
        'ATG', 'AIA', 'YEM', 'SEN', 'BWA', 'MDG', 'BEN', 'BIH', 'BLZ', 'BRB', 
        'CRI', 'DMA', 'GHA', 'GIN', 'GRD', 'HND', 'HUN', 'KAZ', 'KEN', 'KIR', 
        'KNA', 'LCA', 'MNG', 'MSR', 'MWI', 'NAM', 'NER', 'NGA', 'PAN', 'RWA', 
        'SLE', 'SLV', 'SSD', 'THA', 'TTO', 'UKR', 'UZB', 'VCT', 'VGB', 'ZAF', 
        'ZWE', 'BRA'
    ], nargs='+')
    parser.add_argument("--device", help="device", default="cuda:1")
    parser.add_argument("--test", action='store_true')
    parser.add_argument('-e', "--exp_name", default="test_lightning")
    parser.add_argument('--dataset', default="vietnam")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"GLOBAL RANK: {trainer.global_rank}")
    if(trainer.global_rank == 0):
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
    
    'KHM', 'LAO', 'IDN', 'PHL', 'MYS', 'MMR', 'BGD', 'BRN'
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

    test_flag = args.test
    if test_flag:
        #test(c, args.exp_name, args.dataset)
        pass
    else:
        main(c, args.exp_name, args.dataset)