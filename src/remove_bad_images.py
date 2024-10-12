import os

dir = "/mnt/ssd1/agorup/school_mapping/satellite_images/large/VNM/non_school"
for i in range(2000):
    img_path = os.path.join(dir, f"VNM-UNINHABITED-{i}.jpeg")

    if os.path.exists(img_path) and os.path.getsize(img_path) < 10000:
        os.remove(img_path)