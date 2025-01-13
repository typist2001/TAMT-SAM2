import torchvision.transforms.functional as F
import numpy as np
import random
import os
from PIL import Image
from torchvision.transforms import InterpolationMode
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class ToTensor(object):

    def __call__(self, data):
        image, label = data['image'], data['label']
        return {'image': F.to_tensor(image), 'label': F.to_tensor(label)}


class Resize(object):

    def __init__(self, size):
        self.size = size

    def __call__(self, data):
        image, label = data['image'], data['label']

        return {'image': F.resize(image, self.size),
                'label': F.resize(label, self.size, interpolation=InterpolationMode.BICUBIC)}


class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, data):
        image, label = data['image'], data['label']

        if random.random() < self.p:
            return {'image': F.hflip(image), 'label': F.hflip(label)}

        return {'image': image, 'label': label}


class RandomVerticalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, data):
        image, label = data['image'], data['label']

        if random.random() < self.p:
            return {'image': F.vflip(image), 'label': F.vflip(label)}

        return {'image': image, 'label': label}


class Normalize(object):
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        image = F.normalize(image, self.mean, self.std)
        return {'image': image, 'label': label}


class FullDataset(Dataset):
    def __init__(self, image_root, gt_root, size, mode):
        self.images = []
        self.gts = []
        self.text = []

        t1 = "Bright objects, High-contrast objects, Main foreground objects, Distinctive regions, Objects with sharp edges, Areas with strong color contrast, Visually dominant areas"
        t2 = "Blended objects, Background-matching patterns, Objects with camouflage, Low-contrast objects, Regions with subtle texture changes, Hidden shapes in cluttered environments, Objects mimicking surroundings"
        t3 = "Marine animals, Fish with distinctive patterns, Underwater creatures, Sea organisms with unique shapes, Coral and aquatic plants, Ocean wildlife, Fluorescent sea creatures"
        t4 = "Shiny surfaces, Reflective regions, Bright mirrored areas, Objects with glossy finishes, High-gloss materials, Light-bouncing surfaces, Mirror-like reflections"
        t5 = "Polyps in medical images, Irregular growths on tissue, Smooth or lobulated lesions, Abnormal structures in scans, Rounded or oval shapes, Bright or dark regions against normal tissue, Polyp outlines with soft edges, Contrast-enhanced abnormal regions"
        t6=  "Outdoor shadows under sunlight, Shadows created by artificial lights, Urban shadows cast by buildings or trees,Faint shadows blending into textured surfaces, Shadows overlapping with dark regions, Complex interactions between shadow edges and object boundaries"
        # t1="Salient"
        # t2="camouflaged"
        # t3="marine"
        # t4="mirrors"
        # t5="polyps"


        # 加载图像路径和GT路径
        for subfolder in os.listdir(image_root):
            subfolder_path = os.path.join(image_root, subfolder)
            if os.path.isdir(subfolder_path):
                # 加载子文件夹中的图片
                image_paths = [
                    os.path.join(subfolder_path, f)
                    for f in os.listdir(subfolder_path)
                    if f.endswith('.jpg') or f.endswith('.png')
                ]
                self.images.extend(image_paths)
                # 根据文件夹名添加对应的标签
                label = subfolder  # 将文件夹名统一为大写
                if label == "SOD":
                    self.text.extend([t1] * len(image_paths))
                if label == "COD":
                    self.text.extend([t2]* len(image_paths))
                if label == "MAS":
                    self.text.extend([t3] * len(image_paths))
                if label == "MD":
                    self.text.extend([t4] * len(image_paths))
                if label == "PS":
                    self.text.extend([t5] * len(image_paths))
                # if label == "SD":
                #     self.text.extend([t6] * len(image_paths))

        # 加载GT路径
        for subfolder in os.listdir(gt_root):
            subfolder_path = os.path.join(gt_root, subfolder)
            if os.path.isdir(subfolder_path):
                # 加载子文件夹中的GT图片
                self.gts.extend([
                    os.path.join(subfolder_path, f)
                    for f in os.listdir(subfolder_path)
                    if f.endswith('.jpg') or f.endswith('.png')
                ])

        # 确保图像、GT和文本一一对应
        data = list(zip(self.images,  self.text))
        data.sort(key=lambda x: x[0])
        self.gts=sorted(self.gts)

        # 按照图像路径排序，确保它们对应
        # 解包排序后的数据
        self.images, self.text = zip(*data)

        # 转换为列表（如果需要的话）
        self.images = list(self.images)
        self.gts = list(self.gts)
        self.text = list(self.text)

        print(len(self.images))
        print(len(self.gts))
        print(len(self.text))
        if mode == 'train':
            self.transform = transforms.Compose([
                Resize((size, size)),
                RandomHorizontalFlip(p=0.5),
                RandomVerticalFlip(p=0.5),
                ToTensor(),
                Normalize()
            ])
        else:
            self.transform = transforms.Compose([
                Resize((size, size)),
                ToTensor(),
                Normalize()
            ])

    def __getitem__(self, idx):
        text=self.text[idx]
        image = self.rgb_loader(self.images[idx])
        label = self.binary_loader(self.gts[idx])
        data = {'image': image, 'label': label}
        data = self.transform(data)
        data.update({'text':text})

        return data

    def __len__(self):
        return len(self.images)

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')


class TestDataset:
    def __init__(self, image_root, gt_root, size):
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.png')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
        self.gt_transform = transforms.ToTensor()
        self.size = len(self.images)
        self.index = 0

    def load_data(self):
        image = self.rgb_loader(self.images[self.index])
        image = self.transform(image).unsqueeze(0)

        gt = self.binary_loader(self.gts[self.index])
        gt = np.array(gt)

        name = self.images[self.index].split('/')[-1]

        self.index += 1
        return image, gt, name

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')