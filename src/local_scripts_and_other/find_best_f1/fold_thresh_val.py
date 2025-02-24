import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyproj import Transformer

def parse_coordinates(lon_str, lat_str):
    lon = float(re.sub('[^0-9.]', '', lon_str)) * (-1 if 'S' in lon_str else 1)
    lat = float(re.sub('[^0-9.]', '', lat_str)) * (-1 if 'W' in lat_str else 1)
    
    return (lon, lat)

def calculate_metrics(df_preds, df_schools, thresh, verbose=False):
    # Initialize FP
    fp = 0

    # Initialize covered schools set
    covered_schools = set()

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
            fp += 1

    # Calculate TP and FN
    tp = len(covered_schools)
    fn = len(df_schools.index) - len(covered_schools)

    # Calculate precision, recall and F1 score
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    if verbose:
        print(f'Precision: {precision * 100:.2f}%')
        print(f'Recall: {recall * 100:.2f}%')
        print(f'F1 score: {f1 * 100:.2f}%')

    return precision, recall, f1

def main():
    # Load data
    df_preds = pd.read_csv('./inference_vietnam_exhaustive_filtered_ensembling_rotation_mean_NMS.csv')
    df_schools = pd.read_csv('./school_list_wards_3857.csv')

    # Transform preds to 3857
    pyproj_transformer = Transformer.from_crs('EPSG:4326', 'EPSG:3857', always_xy=True)
    df_preds[['longitude', 'latitude']] = df_preds.apply(lambda row: parse_coordinates(row['lon'], row['lat']), axis=1, result_type='expand')
    df_preds['lon'], df_preds['lat'] = pyproj_transformer.transform(df_preds['longitude'], df_preds['latitude'])

    # Split into folds based on ward value
    df_preds['ward'] = df_preds.apply(lambda row: row['image'].split('-')[0], axis=1)
    folds = [df_preds[df_preds['ward'] == ward] for ward in df_preds['ward'].unique()]
    folds_schools = [df_schools[df_schools['ward'] == ward] for ward in df_preds['ward'].unique()]

    # Initialize figures and axes
    figs = [plt.figure(figsize=(10,6)) for _ in range(3)]
    axes = [figs[i].add_subplot() for i in range(3)]

    # Initialize thresholds and metrics list
    final_thresholds = []
    final_precisions = []
    final_recalls = []
    final_f1_scores = []

    for i in range(len(folds)):
        # Get current fold, as well as merge other folds
        df_curr_fold = folds[i]
        df_other_folds = pd.concat([folds[j] for j in range(len(folds)) if j != i])

        df_curr_schools = folds_schools[i]
        df_other_schools = pd.concat([folds_schools[j] for j in range(len(folds)) if j != i])

        # Get current ward name
        curr_ward = df_curr_fold['ward'].unique()[0]

        # Initialize thresholds and metrics
        threshs = []
        precisions = []
        recalls = []
        f1_scores = []

        for i, row in df_curr_fold.iterrows():
            # Get current threshold
            thresh = row['pred']

            # Calculate metrics for current threshold
            precision, recall, f1 = calculate_metrics(df_curr_fold.copy(), df_curr_schools.copy(), thresh, verbose=False)

            # Update thresholds and metrics
            threshs.insert(0, thresh)
            precisions.insert(0, precision)
            recalls.insert(0, recall)
            f1_scores.insert(0, f1)

        # Plot metrics
        axes[0].plot(threshs, precisions, label=curr_ward)
        axes[1].plot(threshs, recalls, label=curr_ward)
        axes[2].plot(threshs, f1_scores, label=curr_ward)

        # Get best F1 score index
        best_f1_index = np.argmax(f1_scores)

        # Get threshold and metrics for best F1 score index
        threshold_final = threshs[best_f1_index]
        precision_final = precisions[best_f1_index]
        recall_final = recalls[best_f1_index]
        f1_final = f1_scores[best_f1_index]

        # Append threshold and metrics
        final_thresholds.append(threshold_final)
        final_precisions.append(precision_final)
        final_recalls.append(recall_final)
        final_f1_scores.append(f1_final)

        # Print final threshold and metrics
        print(f'Current ward: {curr_ward} ({len(df_curr_fold)} predictions, {len(df_curr_schools)} schools)')
        print(f'Final threshold: {threshold_final * 100:.4f}%')

        print(f'------------------------------')
        print(f'Final precision: {precision_final * 100:.4f}%')
        print(f'Final recall: {recall_final * 100:.4f}%')
        print(f'Final F1 score: {f1_final * 100:.4f}%')
        print(f'------------------------------')

        # Get metrics for 0.5 threshold
        precision_05, recall_05, f1_05 = calculate_metrics(df_curr_fold.copy(), df_curr_schools.copy(), 0.5, verbose=False)

        print(f'Precision (0.5 threshold): {precision_05 * 100:.4f}%')
        print(f'Recall (0.5 threshold): {recall_05 * 100:.4f}%')
        print(f'F1 score (0.5 threshold): {f1_05 * 100:.4f}%')
        print(f'------------------------------')

        # Get metrics for other folds
        precision_other_folds, recall_other_folds, f1_other_folds = calculate_metrics(df_other_folds.copy(), df_other_schools.copy(), threshold_final, verbose=False)

        print(f'Other wards performance')
        print(f'Precision: {precision_other_folds * 100:.4f}%')
        print(f'Recall: {recall_other_folds * 100:.4f}%')
        print(f'F1 score: {f1_other_folds * 100:.4f}%')
        print(f'==============================\n')

    # Print threshold and metrics mean +- stddev values
    print('Final results (mean +- stddev)')
    print(f'Threshold: {(np.mean(final_thresholds) * 100):.2f} +- {(np.std(final_thresholds) * 100):.2f}%')
    print(f'Precision: {(np.mean(final_precisions) * 100):.2f} +- {(np.std(final_precisions) * 100):.2f}%')
    print(f'Recall: {(np.mean(final_recalls) * 100):.2f} +- {(np.std(final_recalls) * 100):.2f}%')
    print(f'F1 score: {(np.mean(final_f1_scores) * 100):.2f} +- {(np.std(final_f1_scores) * 100):.2f}%')

    # Add titles, axis labels and legends
    for i, metric in enumerate(['Precision', 'Recall', 'F1 score']):
        axes[i].set_title(f'{metric} comparison for specific thresholds')
        axes[i].set_xlabel('Threshold')
        axes[i].set_ylabel(metric)
        axes[i].legend()

    # Save figures
    figs[0].savefig('./precision_comparison.png')
    figs[1].savefig('./recall_comparison.png')
    figs[2].savefig('./f1_comparison.png')

if __name__ == '__main__':
    main()