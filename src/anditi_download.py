import os
import time
import math
import random
import logging
import argparse
from tqdm import tqdm
import numpy as np

import pandas as pd
import geopandas as gpd
from owslib.wms import WebMapService
import sys

sys.path.insert(0, "../utils/")
import data_utils
import config_utils

import threading
from queue import Queue
import time

from download_satellite_images import producer, consumer

import pickle

SEED = 42
logging.basicConfig(level=logging.INFO)


def download_sat_images(
    creds,
    config,
    category=None,
    iso=None,
    sample_size=None,
    src_crs="EPSG:4326",
    id_col="UID",
    name="clean",
    data=None,
    filename=None,
    out_dir=None,
    download_validated=False
):
    """
    Downloads satellite images based on geographic data points.

    Args:
    - creds (dict): Credentials for accessing the satellite image service.
    - config (dict): Configuration settings.
    - category (str): Type of data category.
    - iso (str, optional): ISO code for the country. Defaults to None.
    - sample_size (int, optional): Number of samples to consider. Defaults to None.
    - src_crs (str, optional): Source coordinate reference system. Defaults to "EPSG:4326".
    - id_col (str, optional): Column name containing unique identifiers. Defaults to "UID".
    - name (str, optional): Name of the dataset. Defaults to "clean".
    - filename (str, optional): File name to load the data. Defaults to None.

    Returns:
    - None
    """
    
    f = "/mnt/ssd1/agorup/school_mapping/inference_data/Anditi_filtered_schools_2-3857.csv"
    data = pd.read_csv(f)
    dest_dir = '/mnt/ssd1/agorup/school_mapping/satellite_images/anditi/large'
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    tasks = []
    for index, row in data.iterrows():
        image_file = f"{dest_dir}/{index}.jpeg"
        bbox = (
                    row["longitude"] - 150 * math.sqrt(2),
                    row["latitude"] - 150 * math.sqrt(2),
                    row["longitude"] + 150 * math.sqrt(2),
                    row["latitude"] + 150 * math.sqrt(2),
                ) 
        tasks.append((image_file, bbox))

        # if category == "school":
        #     image_file_shifted_1 = f"{dest_dir}/{iso}/{category}/shifted/{data[id_col][index]}_SHIFTED-1.jpeg"
        #     bbox_shifted_1 = (
        #         bbox[0] + 100,
        #         bbox[1] + 100,
        #         bbox[2] + 100,
        #         bbox[3] + 100
        #     )
        #     tasks.append((image_file_shifted_1, bbox_shifted_1))

        #     image_file_shifted_2 = f"{dest_dir}/{iso}/{category}/shifted/{data[id_col][index]}_SHIFTED-2.jpeg"
        #     bbox_shifted_2 = (
        #         bbox[0] + 100,
        #         bbox[1] - 100,
        #         bbox[2] + 100,
        #         bbox[3] - 100
        #     )
        #     tasks.append((image_file_shifted_2, bbox_shifted_2))

        #     image_file_shifted_3 = f"{dest_dir}/{iso}/{category}/shifted/{data[id_col][index]}_SHIFTED-3.jpeg"
        #     bbox_shifted_3 = (
        #         bbox[0] - 100,
        #         bbox[1] - 100,
        #         bbox[2] - 100,
        #         bbox[3] - 100
        #     )
        #     tasks.append((image_file_shifted_3, bbox_shifted_3))

        #     image_file_shifted_4 = f"{dest_dir}/{iso}/{category}/shifted/{data[id_col][index]}_SHIFTED-4.jpeg"
        #     bbox_shifted_4 = (
        #         bbox[0] - 100,
        #         bbox[1] + 100,
        #         bbox[2] - 100,
        #         bbox[3] + 100
        #     )
        #     tasks.append((image_file_shifted_4, bbox_shifted_4))
    task_queue = Queue()

    # Number of consumer threads
    num_consumer_threads = 100

    start = time.time()

    # Start producer thread
    producer_thread = threading.Thread(target=producer, args=(task_queue, tasks))
    producer_thread.start()

    # Start consumer threads
    consumer_threads = [
        threading.Thread(target=consumer, args=(task_queue,)) for _ in range(num_consumer_threads)
    ]
    for thread in consumer_threads:
        thread.start()

    # Wait for all tasks to be processed
    producer_thread.join()  # Ensure producer finishes enqueueing all tasks

    for _ in range(num_consumer_threads):
        task_queue.put(None)

    # Wait for all consumer threads to complete
    for thread in consumer_threads:
        thread.join()

    end = time.time()

    print(f'Total time for {len(tasks)} images:', end - start)
    print('Time per image:', (end - start) / len(tasks))



def main():
    # Parser
    parser = argparse.ArgumentParser(description="Satellite Image Download")
    parser.add_argument("--config", help="Config file", default="configs/sat_configs/sat_config_500x500_60cm.yaml")
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

    # vectors_dir = config["vectors_dir"]
    # filename = "clean.geojson"
    # filename = os.path.join(cwd, vectors_dir, "non_school", filename)
    # data = gpd.read_file(filename).reset_index(drop=True)

    iso_codes = [
        'ATG', 'AIA', 'YEM', 'SEN', 'BWA', 'MDG', 'BEN', 'BIH', 'BLZ', 'BRB', 
        'CRI', 'DMA', 'GHA', 'GIN', 'GRD', 'HND', 'HUN', 'KAZ', 'KEN', 'KIR', 
        'KNA', 'LCA', 'MNG', 'MSR', 'MWI', 'NAM', 'NER', 'NGA', 'PAN', 'RWA', 
        'SLE', 'SLV', 'SSD', 'THA', 'TTO', 'UKR', 'UZB', 'VCT', 'VGB', 'ZAF', 
        'ZWE', 'BRA',

        'VNM', 'KHM', 'LAO', 'IDN', 'PHL', 'MYS', 'MMR', 'BGD', 'BRN'
    ]
    #iso_codes = [ "KEN" ]

    try:
        download_sat_images(creds, config)
    except Exception as e:
        print(f"error: {e}")

    



if __name__ == "__main__":
    main()
