import numpy as np
import pandas as pd
import pickle

f = open("/mnt/ssd1/agorup/school_mapping/inference_data/district_name_to_bboxes-overlapping_3857.pkl","rb")
data = pickle.load(f)

adjacency_data = {}
for district in data:
    print(f"district: {district}")
    for index1 in range(len(data[district])):
        ref = data[district][index1]
        ref_lon = (ref["min_lon"] + ref["max_lon"]) / 2
        ref_lat = (ref["min_lat"] + ref["max_lat"]) / 2
        adjacent_tiles = []
        for index2 in range(len(data[district])):
            if index1 == index2:
                continue
            test = data[district][index2]
            test_lon = (test["min_lon"] + test["max_lon"]) / 2
            test_lat = (test["min_lat"] + test["max_lat"]) / 2

            distance = np.sqrt(pow(ref_lon-test_lon,2)+pow(ref_lat-test_lat,2))
            if(distance < 250):
                adjacent_tiles.append(f"{district}-{index2}")
        adjacency_data[f"{district}-{index1}"] = adjacent_tiles

with open("/mnt/ssd1/agorup/school_mapping/inference_data/adjacency_data_overlapping.pkl", "wb") as handle:
    pickle.dump(adjacency_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

print("done")


