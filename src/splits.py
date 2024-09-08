import os
import pandas as pd
import geopandas as gpd
import argparse

sm_dir = "/mnt/sdb/agorup/school_mapping"
splits_dir = '/mnt/sdb/agorup/school_mapping/splits'

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data Cleaning Pipeline")
    parser.add_argument("-n", "--name", default="highres_old_train")
    args = parser.parse_args()

    name = args.name
    cwd = os.path.dirname(os.getcwd())
    data_path = os.path.join(cwd, "data", "vectors", "train", f"{name}.geojson")

    data = gpd.read_file(data_path).reset_index(drop=True)

    data_clean = data[data["clean"] == 0]

    data_train = data_clean[data_clean["dataset"] == "train"]
    data_test = data_clean[data_clean["dataset"] == "test"]


    splits_dir = os.path.join(splits_dir, name)
    if not os.path.exists(splits_dir):
        os.makedirs(splits_dir)

    train_file = open(os.path.join(splits_dir, "train.txt"), 'w')
    val_file = open(os.path.join(splits_dir, "val.txt"), 'w')

    for ind in data_train.index:
        img_path = f"satellite_images/{data_train['iso'][ind]}/{data_train['class'][ind]}/{data_train['UID'][ind]}.jpeg"

        if not os.path.exists(os.path.join(sm_dir, img_path)):
            continue
            img_path = f"satellite_images/{data_train['iso'][ind]}/{data_train['class'][ind]}/lowres/{data_train['UID'][ind]}.jpeg"

        train_file.write(f"{img_path}\n")

    for ind in data_test.index:
        img_path = f"satellite_images/{data_test['iso'][ind]}/{data_test['class'][ind]}/{data_test['UID'][ind]}.jpeg"

        if not os.path.exists(os.path.join(sm_dir, img_path)):
            continue
            img_path = f"satellite_images/{data_test['iso'][ind]}/{data_test['class'][ind]}/lowres/{data_test['UID'][ind]}.jpeg"

        val_file.write(f"{img_path}\n")

    train_file.close()
    val_file.close()
