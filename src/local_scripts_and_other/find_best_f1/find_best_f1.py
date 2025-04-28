import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyproj import Transformer

def parse_coordinates(lon_str, lat_str):
    lon = float(re.sub('[^0-9.]', '', lon_str)) * (-1 if 'S' in lon_str else 1)
    lat = float(re.sub('[^0-9.]', '', lat_str)) * (-1 if 'W' in lat_str else 1)
    
    return (lon, lat)

def calculate_metrics(df_preds, df_schools, thresh, top_n_fp=10, top_n_hardest=10, verbose=False):
    # Initialize FP
    fp = 0

    # Initialize covered schools set
    covered_schools = set()

    # Initialize false positives list
    false_positives = []

    # Iterate over preds
    for i, row in df_preds.iterrows():
        # Check if prediction is under threshold
        if row['pred'] < thresh:
            continue

        # Get lon, lat
        lon, lat = row['lon'], row['lat']
        
        # Find schools which are covered by this prediction
        df_schools['distance'] = np.sqrt((df_schools['lon'] - lon) ** 2 + (df_schools['lat'] - lat) ** 2)
        contained_schools = df_schools[df_schools['distance'] < 250]

        # Update covered schools or FP counter
        if len(contained_schools.index):
            covered_schools.update(contained_schools.index)
        else:
            false_positives.append((row['image'], row['pred']))
            fp += 1

    # Calculate TP and FN
    tp = len(covered_schools)
    fn = len(df_schools.index) - len(covered_schools)

    # Calculate precision, recall and F1 score
    precision = tp / (tp + fp) if (tp + fp > 0) else 0
    recall = tp / (tp + fn) if (tp + fn > 0) else 0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall > 0) else 0

    # Get top N false positives
    top_n_false_positives_unique = sorted(set(false_positives), key=lambda x: x[1], reverse=True)[:top_n_fp]
    top_n_false_positives = [(img, pred, false_positives.count((img, pred))) for (img, pred) in top_n_false_positives_unique]

    # Initialize hardest schools list
    hardest_schools = []

    # Iterate over schools
    for i, row in df_schools.iterrows():
        # Get lon, lat
        lon, lat = row['lon'], row['lat']
        
        # Find predictions which cover this school
        df_preds['distance'] = np.sqrt((df_preds['lon'] - lon) ** 2 + (df_preds['lat'] - lat) ** 2)
        covering_preds = df_preds[df_preds['distance'] < 250]

        if len(covering_preds.index) == 0:
            # School isn't covered
            continue
        else:
            # Get row with highest pred value
            max_pred_row = covering_preds.loc[covering_preds['pred'].idxmax()]
            hardest_schools.append((max_pred_row['image'], max_pred_row['pred']))

    # Get top N hardest schools
    top_n_hardest_schools_unique = sorted(set(hardest_schools), key=lambda x: x[1])[:top_n_hardest]
    top_n_hardest_schools = [(img, pred, hardest_schools.count((img, pred))) for (img, pred) in top_n_hardest_schools_unique]
        
    if verbose:
        print(f'Precision: {precision * 100:.2f}%')
        print(f'Recall: {recall * 100:.2f}%')
        print(f'F1 score: {f1 * 100:.2f}%')
        print(f'Top {top_n_fp} FP:\n{top_n_false_positives}')
        print(f'Top {top_n_hardest} hardest schools:\n{top_n_hardest_schools}')

    return precision, recall, f1, top_n_false_positives, top_n_hardest_schools

def main():
    # Load data
    df_preds = pd.read_csv('./inference_vietnam_exhaustive_filtered_ensembling_rotation_mean_NMS.csv')
    df_schools = pd.read_csv('./school_list_3857.csv')

    # Transform preds to 3857
    pyproj_transformer = Transformer.from_crs('EPSG:4326', 'EPSG:3857', always_xy=True)
    df_preds[['longitude', 'latitude']] = df_preds.apply(lambda row: parse_coordinates(row['lon'], row['lat']), axis=1, result_type='expand')
    df_preds['lon'], df_preds['lat'] = pyproj_transformer.transform(df_preds['longitude'], df_preds['latitude'])

    # Initialize thresholds and metrics
    threshs = []
    precisions = []
    recalls = []
    f1_scores = []
    top_n_false_positives_list = []
    top_n_hardest_schools_list = []

    top_n_fp = 5
    top_n_hardest = 5

    for i, row in df_preds.iterrows():
        # Get current threshold
        thresh = row['pred']

        # Calculate metrics for current threshold
        precision, recall, f1, top_n_false_positives, top_n_hardest_schools = calculate_metrics(df_preds.copy(), df_schools.copy(), thresh, top_n_fp, top_n_hardest, verbose=False)

        # Update thresholds and metrics
        threshs.insert(0, thresh)
        precisions.insert(0, precision)
        recalls.insert(0, recall)
        f1_scores.insert(0, f1)
        top_n_false_positives_list.insert(0, top_n_false_positives)
        top_n_hardest_schools_list.insert(0, top_n_hardest_schools)

    # Plot precision
    plt.figure()
    plt.plot(threshs, precisions)
    plt.xlabel('Threshold')
    plt.ylabel('Precision')
    plt.title('Precision for specific thresholds')
    plt.savefig('./precision.png')

    # Plot recall
    plt.figure()
    plt.plot(threshs, recalls)
    plt.xlabel('Threshold')
    plt.ylabel('Recall')
    plt.title('Recall for specific thresholds')
    plt.savefig('./recall.png')

    # Plot F1 score
    plt.figure()
    plt.plot(threshs, f1_scores)
    plt.xlabel('Threshold')
    plt.ylabel('F1 Score')
    plt.title('F1 Score for specific thresholds')
    plt.savefig('./f1.png')

    # Get best F1 score index
    best_f1_index = np.argmax(f1_scores)

    # Get threshold and metrics for best F1 score index
    threshold_final = threshs[best_f1_index]
    precision_final = precisions[best_f1_index]
    recall_final = recalls[best_f1_index]
    f1_final = f1_scores[best_f1_index]
    top_n_false_positives_final = top_n_false_positives_list[best_f1_index]
    top_n_hardest_schools_final = top_n_hardest_schools_list[best_f1_index]

    # Print final threshold and metrics
    print(f'Final threshold: {threshold_final * 100:.4f}%')
    print(f'Final precision: {precision_final * 100:.4f}%')
    print(f'Final recall: {recall_final * 100:.4f}%')
    print(f'Final F1 score: {f1_final * 100:.4f}%')
    print(f'Final top {top_n_fp} FP:\n{top_n_false_positives_final}')
    print(f'Final top {top_n_hardest} hardest schools:\n{top_n_hardest_schools_final}')

    # Get metrics for 0.5 threshold
    precision_05, recall_05, f1_05, top_n_false_positives_05, top_n_hardest_schools_05 = calculate_metrics(df_preds.copy(), df_schools.copy(), 0.5, top_n_fp, top_n_hardest, verbose=False)

    print(f'\nPrecision (0.5 threshold): {precision_05 * 100:.4f}%')
    print(f'Recall (0.5 threshold): {recall_05 * 100:.4f}%')
    print(f'F1 score (0.5 threshold): {f1_05 * 100:.4f}%')
    print(f'Top {top_n_fp} FP (0.5 threshold):\n{top_n_false_positives_05}')
    print(f'Top {top_n_hardest} hardest schools (0.5 threshold):\n{top_n_hardest_schools_05}')

if __name__ == '__main__':
    main()