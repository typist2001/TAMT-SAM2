import argparse
import os
import torch
import imageio
import numpy as np
import torch.nn.functional as F
from SAM2Unet_MoE2 import SAM2UNet
from dataset import TestDataset

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, required=True,
                    help="path to the checkpoint of sam2-unet")
parser.add_argument("--test_image_path", type=str, required=True,
                    help="path to the image files for testing")
parser.add_argument("--test_gt_path", type=str, required=True,
                    help="path to the mask files for testing")
parser.add_argument("--save_path", type=str, required=True,
                    help="path to save the predicted masks")
args = parser.parse_args()

t1 = "Bright objects, High-contrast objects, Main foreground objects, Distinctive regions, Objects with sharp edges, Areas with strong color contrast, Visually dominant areas"
t2 = "Blended objects, Background-matching patterns, Objects with camouflage, Low-contrast objects, Regions with subtle texture changes, Hidden shapes in cluttered environments, Objects mimicking surroundings"
t3 = "Marine animals, Fish with distinctive patterns, Underwater creatures, Sea organisms with unique shapes, Coral and aquatic plants, Ocean wildlife, Fluorescent sea creatures"
t4 = "Shiny surfaces, Reflective regions, Bright mirrored areas, Objects with glossy finishes, High-gloss materials, Light-bouncing surfaces, Mirror-like reflections"
t5 = "Polyps in medical images, Irregular growths on tissue, Smooth or lobulated lesions, Abnormal structures in scans, Rounded or oval shapes, Bright or dark regions against normal tissue, Polyp outlines with soft edges, Contrast-enhanced abnormal regions"
t6 = "Outdoor shadows under sunlight, Shadows created by artificial lights, Urban shadows cast by buildings or trees,Faint shadows blending into textured surfaces, Shadows overlapping with dark regions, Complex interactions between shadow edges and object boundaries"

# t1 = "Salient"
# t2 = "camouflaged"
# t3 = "marine"
# t4 = "mirrors"
# t5 = "polyps"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
test_loader = TestDataset(args.test_image_path, args.test_gt_path, 352)
model = SAM2UNet().to(device)
model.load_state_dict(torch.load(args.checkpoint), strict=True)
model.eval()
model.cuda()
os.makedirs(args.save_path, exist_ok=True)
for i in range(test_loader.size):
    with torch.no_grad():
        image, gt, name = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        image = image.to(device)
        res, _, _ = model(image, t3)
        # fix: duplicate sigmoid
        # res = torch.sigmoid(res)
        res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu()
        res = res.numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        res = (res * 255).astype(np.uint8)
        print("Saving " + name)
        imageio.imsave(os.path.join(args.save_path, name[:-4] + ".png"), res)
