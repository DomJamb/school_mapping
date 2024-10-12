from PIL import Image
from torchvision import models, transforms

if __name__ == "__main__":
    filepath = "/mnt/ssd1/agorup/school_mapping/satellite_images/large/BWA/school/lowres/OSM-BWA-SCHOOL-00000000.jpeg"
    image = Image.open(filepath).convert("RGB")

    t = transforms.Compose(
            [
                #transforms.Resize(size),
                #transforms.RandomApply([transforms.RandomRotation((90, 90))], p=0.5),
                transforms.RandomRotation((0,360)),
                transforms.CenterCrop(500),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
            ]
        )
    
    i = t(image)
    i.save("rotation_test.jpeg")

    # transform1 = transforms.RandomRotation((45,45), expand=False)
    # transform2 = transforms.RandomRotation((45,45), expand=True)
    # image2 = transform1(image)
    # image3 = transform2(image)
    # transform = transforms.CenterCrop(352)
    # image4 = transform(image2)
    # image5 = transform(image3)

    # image2.save("rotation_test_expand.jpeg")
    # image3.save("rotation_test_expand_crop.jpeg")
    # print("a")