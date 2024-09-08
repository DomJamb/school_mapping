import numpy as np
import pandas as pd
import pickle

df1 = pd.read_csv("inference_filtered_ghsl.csv")
df2 = pd.read_csv("inference_vietnam.csv")

df3 = df2

for i, row in df2.iterrows():
    image = row["image"]
    pos = image.rfind('-')
    district = image[:pos]
    index = int(image[pos+1:-5])
    exists = ((df1['district']==district) & (df1['index']==index)).any()
    if not exists:
        df3 = df3[df3["image"] != row["image"]]

df3.to_csv("inference_vietnam_filtered.csv")




