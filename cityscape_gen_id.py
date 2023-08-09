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

from pytorch_lightning import seed_everything
from annotator.util import resize_image, HWC3
from annotator.uniformer import UniformerDetector
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler


dup_num = 1
ddim_steps = 100
strength = 1.0
scale = 9.0  # scale for classifier-free guidance
seed = 42
prompt = "a professional, detailed, clean, high-quality image" 


output_path = os.path.join('./out_data', datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

gt_path = os.path.join(output_path, 'gt')
img_path = os.path.join(output_path, 'img')
size = 512

transform = A.Compose([
    A.Resize(width=2*size, height=size),
    A.RandomCrop(width=size, height=size),
    A.HorizontalFlip(p=0.5)
])

def process(input_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, detect_resolution, ddim_steps, guess_mode, strength, scale, seed, eta):
    with torch.no_grad():
        # input_image = HWC3(input_image)
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

if not os.path.exists(img_path):
    os.makedirs(img_path)


# Load model
apply_uniformer = UniformerDetector()

model = create_model('./models/cldm_v21_id.yaml').cpu()
model.load_state_dict(load_state_dict('./results/cityscapes/23-06-30/lightning_logs/version_0/checkpoints/epoch=49-step=37199.ckpt', location='cuda'))
model = model.cuda()
ddim_sampler = DDIMSampler(model)


# find all file with "_gtFine_color.png" suffix in "./training/cityscapes/gtFine/train/" and its subdirectories, and save the path to a list
ori_color = []
ori_id_mask = []
for root, dirs, files in os.walk("./training/cityscapes/gtFine/train/"):
    for file in files:
        if file.endswith("_gtFine_color.png"):
            ori_color.append(os.path.join(root, file))
        if file.endswith("_gtFine_labelIds.png"):
            ori_id_mask.append(os.path.join(root, file))

assert len(ori_color) == len(ori_id_mask)

for i in range(len(ori_color)):
    for j in range(dup_num):
        color = cv2.imread(ori_color[i])
        id_mask = cv2.imread(ori_id_mask[i])

        # to one-hot
        id_mask = id_mask[:, :, 0]
        id_mask = np.eye(34)[id_mask].astype(np.uint8)

        color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)

        data = transform(image=color, mask=id_mask)
        color, id_mask = data['image'], data['mask']

        # Sample
        print("Sampling: {}/{}".format(i, len(ori_color)))
        control, result = process(id_mask, prompt, "", "", 1, size, size, ddim_steps, False, strength, scale, seed, 0.0)

        # Save
        cv2.imwrite(os.path.join(gt_path, str(i) + "_" + str(j) + ".png"), color)
        cv2.imwrite(os.path.join(img_path, str(i) + "_" + str(j) + ".png"), result[0])

print("Done!")