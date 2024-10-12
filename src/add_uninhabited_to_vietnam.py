
import os
import sys
sys.path.insert(0, "../utils/")
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import rasterio as rio
from rasterio.mask import mask
import numpy as np
import pickle
from tqdm import tqdm


cwd = os.path.dirname(os.getcwd())

def pickleToDataframe(data):
    arr = []
    for district in data:
        for index in range(len(data[district])):
            longitude = (data[district][index]["min_lon"] + data[district][index]["max_lon"]) / 2
            latitude = (data[district][index]["min_lat"] + data[district][index]["max_lat"]) / 2
            arr.append({"district":district, "index":index, "longitude":longitude, "latitude": latitude})
    df = pd.DataFrame(arr)
    return df

def main():    
    f1 = open("/mnt/ssd1/agorup/school_mapping/inference_data/vietnam_bboxes_3857.pkl", 'rb')
    f2 = "/home/agorup/school_mapping/data/vectors/train/vietnam_train.geojson"
    #f_df = "/mnt/ssd1/agorup/school_mapping/inference_data/inference_unfiltered.csv"
    #f2 = "/mnt/ssd1/agorup/school_mapping/inference_data/inference_filtered_ghsl.csv"

    # if(os.path.exists(f2)):
    #     df = pd.read_csv(f2)
    #     data = {}
    #     for i, row in df.iterrows():
    #         p = row["image"].rfind("-")
    #         district = row["image"][:p]
    #         index = row["image"][p+1:-5]


    #else:


    data = pickle.load(f1)
    data = pickleToDataframe(data)
    vietnam_dataset = gpd.read_file(f2)
         
    ghsl_path = "/mnt/ssd1/agorup/school_mapping/rasters/ghsl/GHS_BUILT_C_FUN_E2018_GLOBE_R2023A_54009_10_V1_0.tif"
    
    #data = gpd.GeoDataFrame(data, geometry="geometry", crs="EPSG:3857")

    images_folder = "/mnt/ssd1/agorup/school_mapping/satellite_images/large/VNM/non_school"
    uninhabited_tiles = []
    uninhabited_tiles_df = pd.DataFrame(columns=["UID", "lon", "lat"])

    while len(uninhabited_tiles) < 2000:
        subdata = data.sample(1)
        subdata["geometry"] = subdata.apply(lambda row: Point(row["longitude"], row["latitude"]), axis=1)
        subdata = gpd.GeoDataFrame(subdata, geometry="geometry", crs="EPSG:3857")

        #subdata = subdata.to_crs("EPSG:3857")
        subdata["geometry"] = subdata["geometry"].buffer(150, cap_style=3)

        # Mask the raster data with the buffered geometry from Microsoft
        image = []
        pixel_sum = 0

        with rio.open(ghsl_path) as src:
            subdata = subdata.to_crs("ESRI:54009")
            geometry = [subdata.iloc[0]["geometry"]]
            image, transform = rio.mask.mask(src, geometry, crop=True)
            image[image == 255] = 0 # no pixel value
            pixel_sum = np.sum(image)

        if pixel_sum == 0:
            subdata2 = subdata.to_crs("EPSG:3857")
            uninhabited_tiles_df.loc[len(uninhabited_tiles_df)] = {"UID": f"VNM-UNINHABITED-{len(uninhabited_tiles)}", "lon": subdata2.iloc[0]["longitude"], "lat": subdata2.iloc[0]["latitude"]}
            uninhabited_tiles.append(subdata.iloc[0].values)
            

            if len(uninhabited_tiles) % 50 == 0:
                print(len(uninhabited_tiles))

    value_dict = {
        "UID": [f"VNM-UNINHABITED-{i}" for i in range(len(uninhabited_tiles))],
        "source" : ["random" for i in range(len(uninhabited_tiles))],
        "iso" : ["VNM" for i in range(len(uninhabited_tiles))],
        "country" : ["Viet Nam" for i in range(len(uninhabited_tiles))],
        "region" : ["South-eastern Asia" for i in range(len(uninhabited_tiles))],
        "subregion" : [row[0] for row in uninhabited_tiles],
        "name" : [f"Uninhabited {row[1]}" for row in uninhabited_tiles],
        "giga_id_school" : [None for i in range(len(uninhabited_tiles))],
        "clean" : [0 for i in range(len(uninhabited_tiles))],
        "class" : ["non_school" for i in range(len(uninhabited_tiles))],
        "ghsl_smod" : [1 for i in range(len(uninhabited_tiles))],
        "rurban" : ["rural" for i in range(len(uninhabited_tiles))],
        "dataset" : ["train" for i in range(len(uninhabited_tiles))],
        "geometry" : [row[4] for row in uninhabited_tiles]
    }

    gdf = gpd.GeoDataFrame(value_dict, crs="ESRI:54009")
    gdf = gdf.to_crs("ESRI:54009")
    vietnam_dataset = vietnam_dataset.to_crs("ESRI:54009")
    gdf2 = gpd.GeoDataFrame( pd.concat([vietnam_dataset, gdf], ignore_index=True) )

    gdf2.to_file("/home/agorup/school_mapping/data/vectors/train/vietnam_uninhabited_train.geojson")
    uninhabited_tiles_df.to_csv("uninhabited_tiles_train.csv", index=False)

    # for index in tqdm(range(len(data))):

    #     # Extract a single row from the DataFrame
    #     subdata = data.iloc[[index]]
    #     subdata = subdata.to_crs("EPSG:3857")
    #     subdata["geometry"] = subdata["geometry"].buffer(150, cap_style=3)

    #     # Mask the raster data with the buffered geometry from Microsoft
    #     image = []
    #     pixel_sum = 0
    #     # If no building pixels found, attempt with GHSL data
    #     if pixel_sum == 0:
    #         with rio.open(ghsl_path) as src:
    #             subdata = subdata.to_crs("ESRI:54009")
    #             geometry = [subdata.iloc[0]["geometry"]]
    #             image, transform = rio.mask.mask(src, geometry, crop=True)
    #             image[image == 255] = 0 # no pixel value
    #             pixel_sum = np.sum(image)

    #     # Appending the pixel sum to the list
    #     pixel_sums.append(pixel_sum) 

    # # Filter data based on pixel sums and updating DataFrame accordingly
    # data["sum"] = pixel_sums
    # data = data[data["sum"] > 0]
    # data = data.reset_index(drop=True)

    # data.to_csv(f2)

    

if __name__ == "__main__":
    main()