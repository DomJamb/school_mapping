import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm

f = open("/mnt/sdb/agorup/school_mapping/inference_data/adjacency_data_overlapping.pkl","rb")
adjacency_data = pickle.load(f)

df = pd.read_csv("/home/djambrovic/school_mapping/exp/fine_tune_anditi/gaussian/inference_vietnam_filtered_ensembling_rotation_mean.csv")
df = df[df["pred"]>0.5]
df2 = df.copy()

counter = 0

for i, row in tqdm(df2.iterrows(), total=len(df2)):
    name = row["image"]
    tile = name[:-5]
    
    if name in df["image"].values:
        counter += 1

        indices = []
        adjacent_tiles = adjacency_data[tile]
        for adjacent_tile in adjacent_tiles:
            adjacent_tile_name = f"{adjacent_tile}.jpeg"
            if adjacent_tile_name in df["image"].values:
                #df = df[df["image"] != adjacent_tile_name]
                ind = df[df["image"] == adjacent_tile_name].index
                indices.append(ind.values[0])
        df = df.drop(indices)

df.to_csv("/home/djambrovic/school_mapping/exp/fine_tune_anditi/gaussian/inference_vietnam_filtered_ensembling_rotation_mean_NMS.csv")

print(counter)




