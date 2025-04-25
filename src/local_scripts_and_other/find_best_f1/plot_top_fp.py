import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image
from pyproj import Transformer

device = "cuda:0"
pyproj_transformer = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)

if __name__ == '__main__':
    dest_dir = '/mnt/sdb/agorup/school_mapping/satellite_images/inference_exhaustive_overlap/large'
    f = "/mnt/sdb/agorup/school_mapping/inference_data/exhaustive/inference_overlap_filtered_ghsl.csv"

    top_n_fp = [('Binh Chieu Ward-190.jpeg', 0.9590598493814468), ('Linh Xuan Ward-179.jpeg', 0.9548553600907326), ('Tam Binh Ward-93.jpeg', 0.9258525520563126), ('Hiep Binh Phuoc Ward-73.jpeg', 0.924198053777218), ('Linh Xuan Ward-155.jpeg', 0.9145309552550316), ('Binh Chieu Ward-56.jpeg', 0.9023766815662384), ('Linh Xuan Ward-151.jpeg', 0.9022977724671364), ('Linh Xuan Ward-83.jpeg', 0.8436941877007484), ('Linh Xuan Ward-204.jpeg', 0.8166244104504585), ('Linh Xuan Ward-52.jpeg', 0.7875362411141396)]

    df = pd.read_csv(f)
    images = []
    for i, row in df.iterrows():
        district = row["district"]
        image = f"{district}-{row['index']}.jpeg"
        image_file = f"{dest_dir}/{district}/{row['index']}.jpeg"
        lon = row["longitude"]
        lat = row["latitude"]
        geom = pyproj_transformer.transform(lon,lat)
        geom_lon = f"{geom[0]}E" if geom[0] >= 0 else f"{-1*geom[0]}W"
        geom_lat = f"{geom[1]}N" if geom[1] >= 0 else f"{-1*geom[1]}S"
        images.append({"filepath":image_file, "image":image, "lat":geom_lat, "lon":geom_lon})

    df_images = pd.DataFrame(images)

    plt.figure(figsize=(18,8))
    plt.suptitle('Top 10 False Positives', fontsize=20)
    
    for i, (fp_image, prob) in enumerate(top_n_fp):
        row = df_images[df_images['image'] == fp_image].iloc[0]
        path = row['filepath']
        img = Image.open(path)

        plt.subplot(2,5,i + 1)
        plt.title(f'{fp_image.split('.')[0]}, {prob:.4f}')
        plt.imshow(img)
        plt.axis('off')

    plt.savefig('./top_10_fp.png')