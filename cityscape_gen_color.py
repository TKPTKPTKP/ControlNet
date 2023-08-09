from share import *
import config

import os
import cv2
import einops
import numpy as np
import torch
import random
import datetime
import albumentations as A
from PIL import Image

from pytorch_lightning import seed_everything
from annotator.util import resize_image, HWC3
from annotator.uniformer import UniformerDetector
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler


dup_num = 10
ddim_steps = 100
strength = 1.0
scale = 9.0  # scale for classifier-free guidance
seed = 42
prompt = ""
a_prompt = "a professional, detailed, clean, high-quality image" 


output_path = os.path.join('./out_data', datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

gt_path = os.path.join(output_path, 'gt')
id_path = os.path.join(output_path, 'id')
img_path = os.path.join(output_path, 'img')
size = 512

transform = A.Compose([
    # A.Resize(width=2*size, height=size),
    A.RandomCrop(width=size, height=size),
    A.HorizontalFlip(p=0.5)
])

def process(input_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, detect_resolution, ddim_steps, guess_mode, strength, scale, seed, eta):
    with torch.no_grad():
        input_image = HWC3(input_image)
        # detected_map = apply_uniformer(resize_image(input_image, detect_resolution))
        detected_map = resize_image(input_image, detect_resolution)
        img = resize_image(input_image, image_resolution)
        H, W, C = img.shape

        detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_NEAREST)

        control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
        control = torch.stack([control for _ in range(num_samples)], dim=0)
        control = einops.rearrange(control, 'b h w c -> b c h w').clone()

        if seed == -1:
            seed = random.randint(0, 65535)
        seed_everything(seed)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)]}
        un_cond = {"c_concat": None if guess_mode else [control], "c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples)]}
        shape = (4, H // 8, W // 8)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=True)

        model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)  # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01
        samples, intermediates = ddim_sampler.sample(ddim_steps, num_samples,
                                                     shape, cond, verbose=False, eta=eta,
                                                     unconditional_guidance_scale=scale,
                                                     unconditional_conditioning=un_cond)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        x_samples = model.decode_first_stage(samples)
        x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)

        results = [x_samples[i] for i in range(num_samples)]
    return detected_map, results


# check if output path exists, if not, create it
if not os.path.exists(gt_path):
    os.makedirs(gt_path)

if not os.path.exists(id_path):
    os.makedirs(id_path)

if not os.path.exists(img_path):
    os.makedirs(img_path)


# Load model
apply_uniformer = UniformerDetector()

model = create_model('./models/cldm_v21.yaml').cpu()
model.load_state_dict(load_state_dict('./results/cityscapes/23-07-08/lightning_logs/checkpoints/epoch=49-step=37199.ckpt', location='cuda'))
model = model.cuda()
ddim_sampler = DDIMSampler(model)

label_data = []
for root, dirs, files in os.walk("./training/cityscapes/leftImg8bit/train"):
    for file in files:
        if file.endswith(".png"):
            # cut the last folder of root. e.g. ./training/cityscape/leftImg8bit/train/aachen -> aachen
            root = root.split("/")[-1]

            # cut the prefix of file. e.g. aachen_000000_000019_leftImg8bit.png -> aachen_000000_000019
            file = file.split("_")[0] + "_" + file.split("_")[1] + "_" + file.split("_")[2]
            label_data.append((root, file))

# # find all file with "_gtFine_color.png" suffix in "./training/cityscapes/gtFine/train/" and its subdirectories, and save the path to a list
# ori_gt = []
# ori_id = []
# for root, dirs, files in os.walk("./training/cityscapes/gtFine/train/"):
#     for file in files:
#         if file.endswith("_gtFine_color.png"):
#             ori_gt.append(os.path.join(root, file))
#         if file.endswith("_gtFine_color.png"):
#             ori_id.append(os.path.join(root, file))

for label_path in label_data:
    for i in range(dup_num):
        gt = cv2.imread(os.path.join("./training/cityscapes/gtFine/train", label_path[0], label_path[1] + "_gtFine_color.png"))
        gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)

        id = Image.open(os.path.join("./training/cityscapes/gtFine/train", label_path[0], label_path[1] + "_gtFine_labelIds.png"))
        id = np.array(id)

        data = transform(image=gt, mask=id)
        gt, id = data['image'], data['mask']

        # Sample
        print("Sampling: {}/{}".format(label_data.index(label_path), len(label_data)))
        control, result = process(gt, prompt, a_prompt, "", 1, size, size, ddim_steps, False, strength, scale, seed, 0.0)

        # Save
        Image.fromarray(control).save(os.path.join(gt_path, str(label_data.index(label_path)) + "_" + str(i) + ".png"))
        Image.fromarray(id).save(os.path.join(id_path, str(label_data.index(label_path)) + "_" + str(i) + ".png"))
        Image.fromarray(result[0]).save(os.path.join(img_path, str(label_data.index(label_path)) + "_" + str(i) + ".png"))
print("Done!")