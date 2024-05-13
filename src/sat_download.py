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

import threading
from queue import Queue
import time

from download_satellite_images import producer, consumer

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
    cwd = os.path.dirname(os.getcwd())
    
    if data is None:
        if not filename:
            vectors_dir = config["vectors_dir"]
            filename = f"{iso}_{name}.geojson"
            filename = os.path.join(cwd, vectors_dir, category, name, filename)
        data = gpd.read_file(filename).reset_index(drop=True)
    
    if "clean" in data.columns:
        data = data[data["clean"] == 0]
    if "validated" in data.columns and not download_validated:
        data = data[data["validated"] == 0]
    if 'iso' in data.columns:
        data = data[data["iso"] == iso].reset_index(drop=True)
    if sample_size:
        data = data.iloc[:sample_size]

    data = data_utils._convert_crs(data, data.crs, config["srs"])
    logging.info(f"Data dimensions: {data.shape}, CRS: {data.crs}")

    if not out_dir:
        out_dir = os.path.join(cwd, config["rasters_dir"], config["maxar_dir"], iso, category)
    out_dir = data_utils._makedir(out_dir)

    #url = f"{config['digitalglobe_url']}connectid={creds['connect_id']}"
    #wms = WebMapService(url, username=creds["username"], password=creds["password"])

    #bar_format = "{l_bar}{bar:20}{r_bar}{bar:-20b}"
    # for index in tqdm(range(len(data)), bar_format=bar_format):
    #     image_file = os.path.join(out_dir, f"{data[id_col][index]}.tiff")
    #     while not os.path.exists(image_file):
    #         try:
    #             bbox = (
    #                 data.lon[index] - config["size"],
    #                 data.lat[index] - config["size"],
    #                 data.lon[index] + config["size"],
    #                 data.lat[index] + config["size"],
    #             )
    #             img = wms.getmap(
    #                 bbox=bbox,
    #                 layers=config["layers"],
    #                 srs=config["srs"],
    #                 size=(config["width"], config["height"]),
    #                 featureProfile=config["featureprofile"],
    #                 coverage_cql_filter=config["coverage_cql_filter"],
    #                 exceptions=config["exceptions"],
    #                 transparent=config["transparent"],
    #                 format=config["format"],
    #             )
    #             with open(image_file, "wb") as file:
    #                 file.write(img.read())
    #         except Exception as e: 
    #             #logging.info(e)
    #             pass

    dest_dir = '/mnt/sdb/agorup/school_mapping/satellite_images'

    tasks = []
    for index in range(len(data)):
        #image_file = f"../{dest_dir}/{data[id_col][index]}.jpeg"
        image_file = f"{dest_dir}/{iso}/{category}/{data[id_col][index]}.jpeg"
        bbox = (
                     data.lon[index] - config["size"],
                     data.lat[index] - config["size"],
                     data.lon[index] + config["size"],
                     data.lat[index] + config["size"],
                ) 
        tasks.append((image_file, bbox))

        if category == "school":
            image_file_shifted_1 = f"{dest_dir}/{iso}/{category}/shifted/{data[id_col][index]}_SHIFTED-1.jpeg"
            bbox_shifted_1 = (
                bbox[0] + 100,
                bbox[1] + 100,
                bbox[2] + 100,
                bbox[3] + 100
            )
            tasks.append((image_file_shifted_1, bbox_shifted_1))

            image_file_shifted_2 = f"{dest_dir}/{iso}/{category}/shifted/{data[id_col][index]}_SHIFTED-2.jpeg"
            bbox_shifted_2 = (
                bbox[0] + 100,
                bbox[1] - 100,
                bbox[2] + 100,
                bbox[3] - 100
            )
            tasks.append((image_file_shifted_2, bbox_shifted_2))

            image_file_shifted_3 = f"{dest_dir}/{iso}/{category}/shifted/{data[id_col][index]}_SHIFTED-3.jpeg"
            bbox_shifted_3 = (
                bbox[0] - 100,
                bbox[1] - 100,
                bbox[2] - 100,
                bbox[3] - 100
            )
            tasks.append((image_file_shifted_3, bbox_shifted_3))

            image_file_shifted_4 = f"{dest_dir}/{iso}/{category}/shifted/{data[id_col][index]}_SHIFTED-4.jpeg"
            bbox_shifted_4 = (
                bbox[0] - 100,
                bbox[1] + 100,
                bbox[2] - 100,
                bbox[3] + 100
            )
            tasks.append((image_file_shifted_4, bbox_shifted_4))

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
    #iso_codes = [ "ATG" ]
    for iso_code in iso_codes:
        try:
            download_sat_images(creds, config, iso=iso_code, category="school")
        except Exception as e:
            print(f"error with iso code {iso_code}: {e}")

    # for iso_code in iso_codes:
    #     try:
    #         download_sat_images(creds, config, iso=iso_code, category="non_school")
    #     except Exception as e:
    #         print(f"error with iso code {iso_code}: {e}")
    



if __name__ == "__main__":
    main()
