import os
import time
import math
import random
import logging
import argparse
from tqdm import tqdm

import pandas as pd
import geopandas as gpd
from owslib.wms import WebMapService
import sys

sys.path.insert(0, "../utils/")
import data_utils
import config_utils

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Satellite Image Download")
    parser.add_argument("--config", help="Config file", default="configs/config_full.yaml")
    parser.add_argument("--creds", help="Credentials file", default="configs/sat_creds.yaml")
    parser.add_argument("--category", help="Category (e.g. school or non_school)", default="school")
    parser.add_argument("--iso", help="ISO code", default="ATG")
    parser.add_argument("--filename", help="Data file", default=None)
    args = parser.parse_args()

    # Load config
    cwd = os.path.dirname(os.getcwd())
    config_file = os.path.join(cwd, args.config)
    creds_file = os.path.join(cwd, args.creds)
    config = config_utils.load_config(config_file)
    creds = config_utils.create_config(creds_file)

    # Download satellite images


    iso_codes = [
        'ATG', 'AIA', 'YEM', 'SEN', 'BWA', 'MDG', 'BEN', 'BIH', 'BLZ', 'BRB', 
        'CRI', 'DMA', 'GHA', 'GIN', 'GRD', 'HND', 'HUN', 'KAZ', 'KEN', 'KIR', 
        'KNA', 'LCA', 'MNG', 'MSR', 'MWI', 'NAM', 'NER', 'NGA', 'PAN', 'RWA', 
        'SLE', 'SLV', 'SSD', 'THA', 'TTO', 'UKR', 'UZB', 'VCT', 'VGB', 'ZAF', 
        'ZWE', 'BRA',
        'VNM', 'KHM', 'LAO', 'IDN', 'PHL', 'MYS', 'MMR', 'BGD', 'BRN'
    ]

    file = open("number_of_images.txt", "w")
 
    

    vectors_dir = config["vectors_dir"]
    filename = os.path.join(cwd, vectors_dir, "school", "osm.geojson")
    filename_clean = os.path.join(cwd, vectors_dir, "school", "clean.geojson")
    data = gpd.read_file(filename).reset_index(drop=True)
    data_clean = gpd.read_file(filename_clean).reset_index(drop=True)
    data_clean = data_clean[data_clean["clean"] == 0]

    file.write("SCHOOL ------------------------------------------ \n")
    file.write(f"\n")
    file.write("Before cleaning ----------------\n")
    file.write(f"TOTAL : {len(data)}\n\n")

    for iso_code in iso_codes:
        file.write(f"{iso_code} : {len(data[data['iso'] == iso_code])}\n")

    file.write(f"\n")
    file.write("After cleaning ----------------\n")
    file.write(f"TOTAL : {len(data_clean)}\n\n")

    for iso_code in iso_codes:
        file.write(f"{iso_code} : {len(data_clean[data_clean['iso'] == iso_code])}\n")

    file.write("\n\n")

    filename = os.path.join(cwd, vectors_dir, "non_school", "osm.geojson")
    filename_clean = os.path.join(cwd, vectors_dir, "non_school", "clean.geojson")
    data = gpd.read_file(filename).reset_index(drop=True)
    data_clean = gpd.read_file(filename_clean).reset_index(drop=True)
    data_clean = data_clean[data_clean["clean"] == 0]

    file.write("NON_SCHOOL ------------------------------------------ \n")
    file.write(f"\n")
    file.write("Before cleaning ----------------\n")
    file.write(f"TOTAL : {len(data)}\n\n")

    for iso_code in iso_codes:
        file.write(f"{iso_code} : {len(data[data['iso'] == iso_code])}\n")

    file.write(f"\n")
    file.write("After cleaning ----------------\n")
    file.write(f"TOTAL : {len(data_clean)}\n\n")

    for iso_code in iso_codes:
        file.write(f"{iso_code} : {len(data_clean[data_clean['iso'] == iso_code])}\n")


    file.close()