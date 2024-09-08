import os
import argparse
import joblib
import pandas as pd
import logging
import torch

import sys
sys.path.insert(0, "../utils/")
import data_utils
import config_utils
import model_utils
import embed_utils
import eval_utils
import wandb

cwd = os.path.dirname(os.getcwd())

def main(iso, config):
    exp_name = f"{iso}-{config['config_name']}"
    #wandb.run.name = exp_name
    results_dir = os.path.join(cwd, config["exp_dir"], exp_name)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    model = embed_utils.load_model(config)
    data = model_utils.load_data(config, attributes=["rurban", "iso"], verbose=True)
    columns = ["iso", "rurban", "dataset", "class"]

    out_dir = os.path.join(config["vectors_dir"], "embeddings")
    embeddings = embed_utils.get_image_embeddings(
        config, data, model, out_dir, in_dir=None, columns=columns, name="test_all"
    )
    embeddings.columns = [str(x) for x in embeddings.columns]
    
    test = embeddings [embeddings.dataset == "test"]
    train = embeddings[embeddings.dataset == "train"]
    logging.info(train.columns)

    logging.info(f"Test size: {test.shape}")
    logging.info(f"Train size: {train.shape}")
    
    target = "class"
    features = [str(x) for x in embeddings.columns[:-len(columns)]]
    classes = list(embeddings[target].unique())
    logging.info(f"No. of features: {len(features)}")
    logging.info(f"Classes: {classes}")

    logging.info("Training model...")
    classes = [0, 1]
    class_map = {'school': 1, 'non_school': 0}
    train[target] = train[target].map(class_map)
    test[target] = test[target].map(class_map)
    cv = model_utils.model_trainer(c, train, features, target)
    logging.info(f"Best estimator: {cv.best_estimator_}")
    logging.info(f"Best CV score: {cv.best_score_}")

    model = cv.best_estimator_
    model.fit(train[features], train[target].values)
    preds = model.predict(test[features])

    model_file = os.path.join(results_dir, f"{iso}-{config['config_name']}.pkl")
    joblib.dump(model, model_file)

    test["pred"] = preds
    #pos_class = config["pos_class"]
    pos_class = 1
    results = eval_utils.save_results(test, target, pos_class, classes, results_dir)

    for rurban in ["urban", "rural"]:
        subresults_dir = os.path.join(results_dir, rurban)
        subtest = test[test.rurban == rurban]
        if len(subtest) == 0:
            continue
        results = eval_utils.save_results(subtest, target, pos_class, classes, subresults_dir, rurban)
    
    if len(config["iso_codes"]) > 1:
        for iso_code in config["iso_codes"]:
            try:
                subresults_dir = os.path.join(results_dir, iso_code)
                subtest = test[test.iso == iso_code]
                if len(subtest) == 0:
                    continue
                results = eval_utils.save_results(
                    subtest, 
                    target, 
                    pos_class, 
                    classes, 
                    subresults_dir, 
                    iso_code
                )
                for rurban in ["urban", "rural"]:
                    subsubresults_dir = os.path.join(subresults_dir, rurban)
                    subsubtest = subtest[subtest.rurban == rurban]
                    if len(subsubtest) == 0:
                        continue
                    results = eval_utils.save_results(
                        subsubtest, 
                        target, 
                        pos_class, 
                        classes, 
                        subsubresults_dir, 
                        f"{iso_code}_{rurban}"
                    )
            except:
                print(f"error with code{iso_code}")

    print()
    print("TRAIN")
    print()
    preds = model.predict(train[features])
    train["pred"] = preds
    #pos_class = config["pos_class"]
    pos_class = 1
    results_dir_train = os.path.join(cwd, config["exp_dir"], exp_name, "train")
    results = eval_utils.save_results(train, target, pos_class, classes, results_dir_train)

    for rurban in ["urban", "rural"]:
        subresults_dir = os.path.join(results_dir_train, rurban)
        subtest = train[train.rurban == rurban]
        if len(subtest) == 0:
            continue
        results = eval_utils.save_results(subtest, target, pos_class, classes, subresults_dir, rurban)
    
    if len(config["iso_codes"]) > 1:
        for iso_code in config["iso_codes"]:
            try:
                subresults_dir = os.path.join(results_dir_train, iso_code)
                subtest = train[train.iso == iso_code]
                if len(subtest) == 0:
                    continue
                results = eval_utils.save_results(
                    subtest, 
                    target, 
                    pos_class, 
                    classes, 
                    subresults_dir, 
                    iso_code
                )
                for rurban in ["urban", "rural"]:
                    subsubresults_dir = os.path.join(subresults_dir, rurban)
                    subsubtest = subtest[subtest.rurban == rurban]
                    if len(subsubtest) == 0:
                        continue
                    results = eval_utils.save_results(
                        subsubtest, 
                        target, 
                        pos_class, 
                        classes, 
                        subsubresults_dir, 
                        f"{iso_code}_{rurban}"
                    )
            except:
                print(f"error with code{iso_code}")
            
            
if __name__ == "__main__":
    # Parser
    parser = argparse.ArgumentParser(description="Model Training")
    parser.add_argument("--model_config", help="Config file", default="configs/model_configs/dinov2_vitb14-LR.yaml")
    parser.add_argument("--iso", help="ISO code", default=[
        'ATG', 'AIA', 'YEM', 'SEN', 'BWA', 'MDG', 'BEN', 'BIH', 'BLZ', 'BRB', 
        'CRI', 'DMA', 'GHA', 'GIN', 'GRD', 'HND', 'HUN', 'KAZ', 'KEN', 'KIR', 
        'KNA', 'LCA', 'MNG', 'MSR', 'MWI', 'NAM', 'NER', 'NGA', 'PAN', 'RWA', 
        'SLE', 'SLV', 'SSD', 'THA', 'TTO', 'UKR', 'UZB', 'VCT', 'VGB', 'ZAF', 
        'ZWE', 'BRA'
    ], nargs='+')
    args = parser.parse_args()

    # Load config
    
    # log_c = {
    #     key: val for key, val in c.items() 
    #     if key is not None
    #     and ('url' not in key) 
    #     and ('dir' not in key)
    #     and ('file' not in key)
    # }
    iso = args.iso[0]
    # log_c["iso_code"] = iso
    # logging.info(log_c)
    
    # wandb.init(
    #     project="UNICEFv1",
    #     config=log_c,
    #     tags=[c["embed_model"], c["model"]]
    # )


    configs = [
        "configs/model_configs/dinov2_vitl14-xgboost.yaml",
        #"configs/model_configs/dinov2_vitl14-LR.yaml",
        #"configs/model_configs/dinov2_vits14-SVC.yaml",
        #"configs/model_configs/dinov2_vitb14-SVC.yaml",
        #"configs/model_configs/dinov2_vitl14-SVC.yaml",
        #"configs/model_configs/esa_foundation_v1-LR.yaml",
        #"configs/model_configs/esa_foundation_v2-LR.yaml",
        #"configs/model_configs/esa_foundation-LR.yaml",
    ]

    for config in configs:
        #try:
            config_file = os.path.join(cwd, config)
            c = config_utils.load_config(config_file)
            c["iso_codes"] = args.iso
            main(iso, c)
        # except:
        #     pass

    