import re
import argparse

import numpy as np
import pandas as pd
from pyproj import Transformer

import matplotlib.pyplot as plt

def parse_coordinates(lon_str, lat_str):
    lon = float(re.sub('[^0-9.]', '', lon_str)) * (-1 if 'S' in lon_str else 1)
    lat = float(re.sub('[^0-9.]', '', lat_str)) * (-1 if 'W' in lat_str else 1)
    
    return (lon, lat)

def compare(anditi_path, preds_path, anditi_save_path, preds_save_path, stats_path):
    thresh = 0.5
    pyproj_transformer = Transformer.from_crs('EPSG:4326', 'EPSG:3857', always_xy=True)

    df_anditi = pd.read_csv(anditi_path)
    df_preds = pd.read_csv(preds_path)

    # Remove nondistrict schools
    with open('./nondistrict_schools.txt', 'r') as f:
        nondistrict_schools = [line.strip() for line in f.readlines()]

    df_anditi = df_anditi[~df_anditi['image'].isin(nondistrict_schools)]

    # Parse string coordinates
    df_preds[['longitude', 'latitude']] = df_preds.apply(lambda row: parse_coordinates(row['lon'], row['lat']), axis=1, result_type='expand')

    # Transform to EPSG:3857
    df_preds['longitude'], df_preds['latitude'] = pyproj_transformer.transform(df_preds['longitude'], df_preds['latitude'])

    # Initialize found_Anditi column
    df_preds['found_Anditi'] = [[] for _ in range(len(df_preds))]

    # Initialize min_probs, max_probs, mean_probs columns
    df_anditi['min_probs'] = [0. for _ in range(len(df_anditi))]
    df_anditi['max_probs'] = [0. for _ in range(len(df_anditi))]
    df_anditi['mean_probs'] = [0. for _ in range(len(df_anditi))]

    # Initialize counter of found Anditi schools
    cnt = 0

    # Initialize closest predictions rows
    closest_preds_rows = []

    # Iterate over Anditi rows
    for i, row in df_anditi.iterrows():
        img = row['image']
        lon, lat = row['lon'], row['lat']
        
        # Find rows which contain the Anditi location
        df_preds['distance'] = np.sqrt((df_preds['longitude'] - lon) ** 2 + (df_preds['latitude'] - lat) ** 2)
        containing_rows = df_preds[df_preds['distance'] < 250]

        if len(containing_rows) > 0:
            # Update found Anditi schools counter
            cnt += 1
        
            # Update found_Anditi column
            for index in containing_rows.index:
                df_preds.at[index, 'found_Anditi'].append(img)

            # Update probs
            df_anditi.at[i, 'min_probs'] = containing_rows['pred'].min()
            df_anditi.at[i, 'mean_probs'] = containing_rows['pred'].mean()
            df_anditi.at[i, 'max_probs'] = containing_rows['pred'].max()
        else:
            # Find closest prediction
            closest_pred = df_preds.loc[df_preds['distance'].idxmin()]

            # Save pair in dataframe
            closest_preds_rows.append({'Anditi_img': img, 'closest_prediction': closest_pred['image'], 'prob': closest_pred['pred'], 'distance': df_preds['distance'].min()})

    # Save changed dataframes
    df_anditi.to_csv(anditi_save_path, index=False)
    df_preds.drop(columns=['longitude', 'latitude', 'distance'], inplace=True)
    df_preds.to_csv(preds_save_path, index=False)

    # Save pair dataframe
    df_closest_preds = pd.DataFrame(closest_preds_rows, columns=['Anditi_img', 'closest_prediction', 'prob', 'distance'])
    df_closest_preds.to_csv('./anditi_closest_preds_pairs_NMS.csv', index=False)

    # Save closest preds probabilities histogram
    plt.hist(np.array(df_closest_preds['prob'], dtype=np.float32))
    plt.xlabel('Probability')
    plt.ylabel('Count')
    plt.title('False negatives probability (NMS)')
    plt.savefig('./anditi_closest_preds_probs_hist_NMS.png')

    # Calculate recall
    results = f'Found {cnt}/{df_anditi.shape[0]} ({(cnt / df_anditi.shape[0]) * 100:.2f}%) Anditi schools\n'
    results += f'Recall (min probs): {((df_anditi["min_probs"] > thresh).sum() / df_anditi.shape[0]) * 100:.2f}%\n'
    results += f'Recall (mean probs): {((df_anditi["mean_probs"] > thresh).sum() / df_anditi.shape[0]) * 100:.2f}%\n'
    results += f'Recall (max probs): {((df_anditi["max_probs"] > thresh).sum() / df_anditi.shape[0]) * 100:.2f}%'

    # Save stats
    with open(stats_path, 'w') as f:
        f.write(results)
    
    print(results)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Anditi dataset and dense inference predictions comparison')

    parser.add_argument('--anditi_path', help='Path to Anditi dataset csv file', default='./anditi_school.csv')

    # parser.add_argument('--inference_path', help='Path to dense inference predictions csv file', default='./inference_vietnam_filtered_ensembling_rotation_mean.csv')
    # parser.add_argument('--anditi_save_path', help='Path to Anditi dataset save csv file', default='./anditi_school_comparison.csv')
    # parser.add_argument('--inference_save_path', help='Path to dense inference predictions save csv file', default='./inference_vietnam_filtered_ensembling_rotation_mean_comparison.csv')
    # parser.add_argument('--stats_path', help='Path to result statistics file', default='./stats_comparison.txt')

    parser.add_argument('--inference_path', help='Path to dense inference predictions csv file', default='./inference_vietnam_filtered_ensembling_rotation_mean_NMS.csv')
    parser.add_argument('--anditi_save_path', help='Path to Anditi dataset save csv file', default='./anditi_school_comparison_NMS.csv')
    parser.add_argument('--inference_save_path', help='Path to dense inference predictions save csv file', default='./inference_vietnam_filtered_ensembling_rotation_mean_comparison_NMS.csv')
    parser.add_argument('--stats_path', help='Path to result statistics file', default='./stats_comparison_NMS.txt')

    args = parser.parse_args()

    compare(args.anditi_path, args.inference_path, args.anditi_save_path, args.inference_save_path, args.stats_path)