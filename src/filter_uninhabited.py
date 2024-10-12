
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
    df = pd.DataFrame(columns=["district", "index", "longitude", "latitude"])
    for district in data:
        for index in range(len(data[district])):
            longitude = (data[district][index]["min_lon"] + data[district][index]["max_lon"]) / 2
            latitude = (data[district][index]["min_lat"] + data[district][index]["max_lat"]) / 2

            df.loc[len(df)] = {"district":district, "index":index, "longitude":longitude, "latitude": latitude}

    return df

def main():    
    # f1 = open("/mnt/ssd1/agorup/school_mapping/inference_data/district_name_to_bboxes_3857-500px.pkl", 'rb')
    # f1f = "/mnt/ssd1/agorup/school_mapping/inference_data/district_name_to_bboxes_3857-500px_filtered_ghsl.pkl"
    # f_df = "/mnt/ssd1/agorup/school_mapping/inference_data/inference_unfiltered.csv"
    # f2 = "/mnt/ssd1/agorup/school_mapping/inference_data/inference_filtered_ghsl.csv"

    f1 = open("/mnt/ssd1/agorup/school_mapping/inference_data/district_name_to_bboxes-overlapping_3857.pkl", 'rb')
    f1f = "/mnt/ssd1/agorup/school_mapping/inference_data/district_name_to_bboxes-overlapping_3857_filtered_ghsl.pkl"
    f_df = "/mnt/ssd1/agorup/school_mapping/inference_data/inference_overlap_unfiltered.csv"
    f2 = "/mnt/ssd1/agorup/school_mapping/inference_data/inference_overlap_filtered_ghsl.csv"

    # if(os.path.exists(f2)):
    #     df = pd.read_csv(f2)
    #     data = {}
    #     for i, row in df.iterrows():
    #         p = row["image"].rfind("-")
    #         district = row["image"][:p]
    #         index = row["image"][p+1:-5]


    #else:

    if(not os.path.exists(f_df)):
        data = pickle.load(f1)
        data = pickleToDataframe(data)
        data.to_csv(f_df)
    else:
        data = pd.read_csv(f_df)
    ghsl_path = "/mnt/ssd1/agorup/school_mapping/rasters/ghsl/GHS_BUILT_C_FUN_E2018_GLOBE_R2023A_54009_10_V1_0.tif"
    data["geometry"] = data.apply(lambda row: Point(row["longitude"], row["latitude"]), axis=1)
    data = gpd.GeoDataFrame(data, geometry="geometry", crs="EPSG:3857")

    pixel_sums = []
    for index in tqdm(range(len(data))):

        # Extract a single row from the DataFrame
        subdata = data.iloc[[index]]
        subdata = subdata.to_crs("EPSG:3857")
        subdata["geometry"] = subdata["geometry"].buffer(150, cap_style=3)

        # Mask the raster data with the buffered geometry from Microsoft
        image = []
        pixel_sum = 0
        # If no building pixels found, attempt with GHSL data
        if pixel_sum == 0:
            with rio.open(ghsl_path) as src:
                subdata = subdata.to_crs("ESRI:54009")
                geometry = [subdata.iloc[0]["geometry"]]
                image, transform = rio.mask.mask(src, geometry, crop=True)
                image[image == 255] = 0 # no pixel value
                pixel_sum = np.sum(image)

        # Appending the pixel sum to the list
        pixel_sums.append(pixel_sum) 

    # Filter data based on pixel sums and updating DataFrame accordingly
    data["sum"] = pixel_sums
    print(len(data))
    data = data[data["sum"] > 0]
    print(len(data))
    data = data.reset_index(drop=True)

    data.to_csv(f2)

    

if __name__ == "__main__":
    main()