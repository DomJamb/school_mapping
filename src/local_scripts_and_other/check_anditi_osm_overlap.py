import numpy as np
import pandas as pd
import geopandas as gpd
import pickle
from tqdm import tqdm

df_anditi = pd.read_csv("anditi_school.csv")
df_osm = gpd.read_file("vietnam_train.geojson")
df_osm = df_osm[df_osm["class"] == "school"]
df_osm = df_osm[df_osm["dataset"] == "test"]
df_inference_tiles = pd.read_csv("inference_filtered_ghsl.csv")
df_inference_preds = pd.read_csv("inference_vietnam_filtered_ghsl_VNM+Anditi.csv")


same_anditi = pd.DataFrame(columns=df_anditi.columns)
same_inference_distance = pd.DataFrame(columns=list(df_inference_preds.columns) + ["anditi_id"])
same_inference_pred = pd.DataFrame(columns=list(df_inference_preds.columns) + ["anditi_id"])

for i1 in tqdm(range(len(df_anditi))):
    row1 = df_anditi.iloc[i1]
    lon = row1["lon"]
    lat = row1["lat"]
    # for i2, row2 in df_osm.iterrows():
    #     geom = row2["geometry"]
    #     d = np.sqrt(pow(lon-geom.x,2) + pow(lat-geom.y,2))
    #     if d < 500:
    #         same.append(row1)
    #         break
    distance = np.sqrt(np.power(lon - df_inference_tiles["longitude"], 2) + np.power(lat - df_inference_tiles["latitude"], 2))
    filtered_inference = df_inference_tiles[distance < 300].copy()
    if len(filtered_inference) > 0 :
        same_anditi.loc[len(same_anditi)] = row1
        row2 = filtered_inference.loc[distance.idxmin()].copy()
        row2 = df_inference_preds[df_inference_preds["image"] == f"{row2['district']}-{row2['index']}.jpeg"].iloc[0].copy()
        row2["anditi_id"] = int(row1.values[0][:-5])
        same_inference_distance.loc[len(same_inference_distance)] = row2

        filtered_inference["image"] = filtered_inference['district'] + "-" + filtered_inference['index'].astype(str) + ".jpeg"
        filtered_inference_preds = df_inference_preds[df_inference_preds["image"].isin(filtered_inference["image"])]
        pred = filtered_inference_preds["pred"]
        row2 = filtered_inference_preds.loc[pred.idxmax()].copy()
        row2["anditi_id"] = int(row1.values[0][:-5])
        same_inference_pred.loc[len(same_inference_pred)] = row2

    
     
    # for i2, row2 in df_inference.iterrows():
    #     lon2 = row2["longitude"]
    #     lat2 = row2["latitude"]
    #     d = np.sqrt(pow(lon-lon2,2) + pow(lat-lat2,2))
    #     if d < 150:
    #         #print("same")
    #         same1.loc[len(same1)] = row1
    #         same2.loc[len(same2)] = row2
    #         break


same_anditi.to_csv("oia_anditi.csv")


df_inference_preds_values = pd.DataFrame(columns=list(df_inference_preds.columns) + ["anditi_id"])
df_inference_distance_values = pd.DataFrame(columns=list(df_inference_preds.columns) + ["anditi_id"])

for i1, row1 in same_inference_pred.iterrows():
    anditi_pred = df_anditi[df_anditi["image"].str[:-5] == str(row1["anditi_id"])]
    if len(anditi_pred) == 0 or anditi_pred.iloc[0]["pred"] < 0.5:
        continue
    image = row1["image"]
    row2 = df_inference_preds[df_inference_preds["image"] == image]
    
    if len(row2) > 0:
        df_inference_preds_values.loc[len(df_inference_preds_values)] = row1

df_inference_preds_values = df_inference_preds_values.sort_values("pred", ascending=False)
df_inference_preds_values.to_csv("oai_inference_VNM+Anditi_max_pred.csv")


for i1, row1 in same_inference_distance.iterrows():
    anditi_pred = df_anditi[df_anditi["image"].str[:-5] == str(row1["anditi_id"])]
    if len(anditi_pred) == 0 or anditi_pred.iloc[0]["pred"] < 0.5:
        continue
    image = row1["image"]
    row2 = df_inference_preds[df_inference_preds["image"] == image]
    
    if len(row2) > 0:
        df_inference_distance_values.loc[len(df_inference_distance_values)] = row1

df_inference_distance_values = df_inference_distance_values.sort_values("pred", ascending=False)
df_inference_distance_values.to_csv("oai_inference_VNM+Anditi_min_distance.csv")

# HARD EXAMPLES
df_inference_preds_values = pd.DataFrame(columns=list(df_inference_preds.columns) + ["anditi_id"])
df_inference_distance_values = pd.DataFrame(columns=list(df_inference_preds.columns) + ["anditi_id"])
for i1, row1 in same_inference_pred.iterrows():
    anditi_pred = df_anditi[df_anditi["image"].str[:-5] == str(row1["anditi_id"])]
    if len(anditi_pred) == 0 or anditi_pred.iloc[0]["pred"] > 0.5:
        continue
    image = row1["image"]
    row2 = df_inference_preds[df_inference_preds["image"] == image]
    
    if len(row2) > 0:
        df_inference_preds_values.loc[len(df_inference_preds_values)] = row1

df_inference_preds_values = df_inference_preds_values.sort_values("pred", ascending=False)
df_inference_preds_values.to_csv("oai_inference_VNM+Anditi_max_pred_hard_examples.csv")


for i1, row1 in same_inference_distance.iterrows():
    anditi_pred = df_anditi[df_anditi["image"].str[:-5] == str(row1["anditi_id"])]
    if len(anditi_pred) == 0 or anditi_pred.iloc[0]["pred"] > 0.5:
        continue
    image = row1["image"]
    row2 = df_inference_preds[df_inference_preds["image"] == image]
    
    if len(row2) > 0:
        df_inference_distance_values.loc[len(df_inference_distance_values)] = row1

df_inference_distance_values = df_inference_distance_values.sort_values("pred", ascending=False)
df_inference_distance_values.to_csv("oai_inference_VNM+Anditi_min_distance_hard_examples.csv")





