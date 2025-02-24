import numpy as np
import pandas as pd

def main():
    # Load data
    df_schools = pd.read_csv('./school_list_3857.csv')
    df_tiles = pd.read_csv('./inference_overlap_filtered_ghsl.csv')

    # Initialize wards list
    wards = []

    for i, row in df_schools.iterrows():
        # Get lon, lat
        lon, lat = row['lon'], row['lat']

        # Calculate distance for each tile
        df_tiles['distance'] = np.sqrt((df_tiles['longitude'] - lon) ** 2 + (df_tiles['latitude'] - lat) ** 2)

        # Get closest tile
        closest_index = df_tiles['distance'].argmin()
        
        # Append closest tile ward
        wards.append(df_tiles.iloc[closest_index]['district'])

    # Add wards list as new column and save updated dataframe
    df_schools['ward'] = wards
    df_schools.to_csv('./school_list_wards_3857.csv', index=False)

if __name__ == '__main__':
    main()