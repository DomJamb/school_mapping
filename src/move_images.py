import os

IMG_PATH = "/mnt/sdb/agorup/school_mapping/satellite_images"

if __name__ == "__main__":

    iso_codes = [
        'ATG', 'AIA', 'YEM', 'SEN', 'BWA', 'MDG', 'BEN', 'BIH', 'BLZ', 'BRB', 
        'CRI', 'DMA', 'GHA', 'GIN', 'GRD', 'HND', 'HUN', 'KAZ', 'KEN', 'KIR', 
        'KNA', 'LCA', 'MNG', 'MSR', 'MWI', 'NAM', 'NER', 'NGA', 'PAN', 'RWA', 
        'SLE', 'SLV', 'SSD', 'THA', 'TTO', 'UKR', 'UZB', 'VCT', 'VGB', 'ZAF', 
        'ZWE', 'BRA'
    ]

    for iso_code in iso_codes:
        if not os.path.exists(os.path.join(IMG_PATH, iso_code)):
            os.makedirs(os.path.join(IMG_PATH, iso_code, "school"))
            os.makedirs(os.path.join(IMG_PATH, iso_code, "non_school"))

    dir_school = os.path.join(IMG_PATH, "school")
    for filename in os.listdir(dir_school):
        file = os.path.join(dir_school, filename)
        if os.path.isfile(file):
            iso_code = filename[4:7]
            os.replace(file, os.path.join(IMG_PATH, iso_code, "school", filename))

    dir_non_school = os.path.join(IMG_PATH, "non_school")
    for filename in os.listdir(dir_non_school):
        file = os.path.join(dir_non_school, filename)
        if os.path.isfile(file):
            iso_code = filename[4:7]
            os.replace(file, os.path.join(IMG_PATH, iso_code, "non_school", filename))
