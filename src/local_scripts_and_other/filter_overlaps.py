import numpy as np
import pandas as pd
import pickle

f = open("adjacency_data.pkl","rb")
adjacency_data = pickle.load(f)

df = pd.read_csv("inference_vietnam_filtered_ensembling_rotation_mean1.csv")
df = df[df["pred"]>0.5]

counter = 0

for tile in adjacency_data:
    name = f"{tile}"
    
    if name in df["image"].values:
        counter += 1

        adjacent_tiles = adjacency_data[tile]
        for adjacent_tile in adjacent_tiles:
            adjacent_tile_name = f"{adjacent_tile}"
            if adjacent_tile_name in df["image"].values:
                df = df[df["image"] != adjacent_tile_name]

print(counter)




