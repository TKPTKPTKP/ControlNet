import cv2
import numpy as np
import os
import albumentations as A

from torch.utils.data import Dataset

transform = A.Compose([
    # A.Resize(width=1024, height=512),
    A.RandomCrop(width=512, height=512),
    A.HorizontalFlip(p=0.5)
])


fix_prompt = "a professional, detailed, high-quality image" 
additional_propmt = "while (111, 74, 0) refers to dynamic, (81, 0, 81) refers to ground, (128, 64, 128) refers to road, \
            (244, 35, 232)sidewalk, (250, 170, 160)parking, (230, 150, 140)rail track, (70, 70, 70)building, \
            (102, 102, 156)wall, (190, 153, 153)fence, (180, 165, 180)guard rail, (150, 100, 100)bridge, (150, 120, 90)tunnel,\
            (153, 153, 153)pole, (250, 170, 30)traffic light, (220, 220, 0)traffic sign, (107, 142, 35)vegetation,\
            (152, 251, 152)terrain, (70, 130, 180)sky, (220, 20, 60)person, (255, 0, 0)rider, (0, 0, 142)car, (0, 0, 70)truck, \
            (0, 60, 100)bus, (0, 0, 90)caravan, (0, 0, 110)trailer, (0, 80, 100)train, (0, 0, 230)motorcycle, (119, 11, 32)bicycle, (0, 0, 142)license plate."

class Cityscape(Dataset):
    def __init__(self, use_id=False):
        self.use_id = use_id
        self.data = []

        for root, dirs, files in os.walk("./training/cityscapes/leftImg8bit/train"):
            for file in files:
                if file.endswith(".png"):
                    # cut the last folder of root. e.g. ./training/cityscape/leftImg8bit/train/aachen -> aachen
                    root = root.split("/")[-1]

                    # cut the prefix of file. e.g. aachen_000000_000019_leftImg8bit.png -> aachen_000000_000019
                    file = file.split("_")[0] + "_" + file.split("_")[1] + "_" + file.split("_")[2]
                    self.data.append((root, file))

        print("Total number of images: ", len(self.data))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        if not self.use_id:  # 用颜色mask当作control
            control_filename = os.path.join("./training/cityscapes/gtFine/train", item[0], item[1] + "_gtFine_color.png")
            image_filename = os.path.join("./training/cityscapes/leftImg8bit/train", item[0], item[1] + "_leftImg8bit.png")
            prompt = fix_prompt

            control = cv2.imread(control_filename)
            image = cv2.imread(image_filename)

            # Do not forget that OpenCV read images in BGR order.
            control = cv2.cvtColor(control, cv2.COLOR_BGR2RGB)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            data = transform(image=image, mask=control)
            image, control = data['image'], data['mask']

            # Normalize source images to [0, 1].
            control = control.astype(np.float32) / 255.0

            # Normalize target images to [-1, 1].
            image = (image.astype(np.float32) / 127.5) - 1.0

            return dict(jpg=image, txt=prompt, hint=control)


        else:  # 用id mask当作control
            control_filename = os.path.join("./training/cityscapes/gtFine/train", item[0], item[1] + "_gtFine_labelIds.png")
            image_filename = os.path.join("./training/cityscapes/leftImg8bit/train", item[0], item[1] + "_leftImg8bit.png")

            prompt = fix_prompt

            control = cv2.imread(control_filename)
            image = cv2.imread(image_filename)

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            data = transform(image=image, mask=control)
            image, control = data['image'], data['mask']

            # to one-hot
            control = control[:, :, 0]
            control = np.eye(34)[control].astype(np.uint8)

            # Normalize target images to [-1, 1].
            image = (image.astype(np.float32) / 127.5) - 1.0

            return dict(jpg=image, txt=prompt, hint=control)