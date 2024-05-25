from PIL import Image
from torchvision import models, transforms

if __name__ == "__main__":
    filepath = "/mnt/sdb/agorup/school_mapping/satellite_images/ATG/school/OSM-ATG-SCHOOL-00000000.jpeg"
    image = Image.open(filepath).convert("RGB")

    transform1 = transforms.RandomRotation((45,45), expand=False)
    transform2 = transforms.RandomRotation((45,45), expand=True)
    image2 = transform1(image)
    image3 = transform2(image)
    transform = transforms.CenterCrop(352)
    image4 = transform(image2)
    image5 = transform(image3)

    image2.save("rotation_test_expand.jpeg")
    image3.save("rotation_test_expand_crop.jpeg")
    print("a")