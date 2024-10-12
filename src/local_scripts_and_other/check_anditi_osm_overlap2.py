import numpy as np
import pandas as pd
import geopandas as gpd
import pickle
from tqdm import tqdm

df1 = pd.read_csv("overlap_anditi_inference_2.csv")
df2 = pd.read_csv("inference_vietnam_overlap.csv")
df_anditi = pd.read_csv("anditi_school.csv")

df_new = pd.DataFrame(columns=list(df2.columns) + ["anditi_id"])

for i1, row1 in df1.iterrows():
    anditi_pred = df_anditi[df_anditi["image"].str[:-5] == str(row1["anditi_id"])]
    if len(anditi_pred) == 0 or anditi_pred.iloc[0]["pred"] < 0.5:
        continue
    image = f"{row1['district']}-{row1['index']}.jpeg"
    row2 = df2[df2["image"] == image]
    
    if len(row2) > 0:
        row2 = row2.iloc[0].copy()
        row2["anditi_id"] = row1["anditi_id"]
        df_new.loc[len(df_new)] = row2

df_new = df_new.sort_values("pred", ascending=False)
df_new.to_csv("overlap_anditi_inference_3.csv")








