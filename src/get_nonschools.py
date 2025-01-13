import pandas as pd

base_path = '/mnt/sdb/agorup/school_mapping/inference_data/exhaustive'
tiles = pd.read_csv(f'{base_path}/inference_overlap_filtered_ghsl.csv')
schools = pd.read_csv(f'{base_path}/school_list_3857.csv')

schools_list = pd.DataFrame(columns=['image'])
nonschools_list = pd.DataFrame(columns=['image'])

for i, row in tiles.iterrows():
    image = f"{row['district']}-{row['index']}.jpeg"

    if any((schools['lon'] >= row['longitude'] - 155) & (schools['lon'] <= row['longitude'] + 155) & 
           (schools['lat'] >= row['latitude'] - 155) & (schools['lat'] <= row['latitude'] + 155)):
        schools_list.loc[len(schools_list)] = {'image': image}
    else:
        nonschools_list.loc[len(nonschools_list)] = {'image': image}

schools_list.to_csv(f'{base_path}/school_list.csv')
nonschools_list.to_csv(f'{base_path}/nonschool_list.csv')

print(f'Number of  schools: {len(schools_list.index)}')
print(f'Number of  nonschools: {len(nonschools_list.index)}')
print(f'Expected number of schools: {len(schools.index)}')

print(f'Total: {len(schools_list.index) + len(nonschools_list.index)}')
print(f'Expected total: {len(tiles.index)}')