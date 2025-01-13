import pandas as pd

base_path = '/mnt/sdb/agorup/school_mapping/inference_data/exhaustive'
tiles = pd.read_csv(f'{base_path}/inference_overlap_filtered_ghsl.csv')
schools = pd.read_csv(f'{base_path}/school_list_3857.csv')

schools_list = pd.DataFrame(columns=['image'])
nonschools_list = pd.DataFrame(columns=['image'])

for i, row in tiles.iterrows():
    image = f"{row['district']}-{row['index']}.jpeg"

    if any((schools['lon'] == row['longitude']) & (schools['lat'] == row['latitude'])):
        schools_list.loc[len(schools_list)] = {'image': image}
    else:
        nonschools_list.loc[len(nonschools_list)] = {'image': image}

schools_list.to_csv(f'{base_path}/school_list.csv')
nonschools_list.to_csv(f'{base_path}/nonschool_list.csv')