import os
import argparse
import random
import numpy as np
import torch
import torch.optim as opt
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
# from dataset import FullDataset
from CoDataset import FullDataset
from SAM2Unet_MoE2 import SAM2UNet
# from clip_SAM2UNet import SAM2UNet
import torch.nn as nn
from torch.utils.data import WeightedRandomSampler
from collections import Counter

parser = argparse.ArgumentParser("SAM2-UNet")
parser.add_argument("--hiera_path", type=str, required=True, 
                    help="path to the sam2 pretrained hiera")
parser.add_argument("--train_image_path", type=str, required=True, 
                    help="path to the image that used to train the model")
parser.add_argument("--train_mask_path", type=str, required=True,
                    help="path to the mask file for training")
parser.add_argument('--save_path', type=str, required=True,
                    help="path to store the checkpoint")
parser.add_argument("--epoch", type=int, default=20, 
                    help="training epochs")
parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
parser.add_argument("--batch_size", default=12, type=int)
parser.add_argument("--weight_decay", default=5e-4, type=float)
args = parser.parse_args()


def structure_loss(pred, mask):
    weit = 1 + 5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit*wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))
    pred = torch.sigmoid(pred)
    inter = ((pred * mask)*weit).sum(dim=(2, 3))
    union = ((pred + mask)*weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1)/(union - inter+1)
    return (wbce + wiou).mean()


t1 = "Bright objects, High-contrast objects, Main foreground objects, Distinctive regions, Objects with sharp edges, Areas with strong color contrast, Visually dominant areas"
t2 = "Blended objects, Background-matching patterns, Objects with camouflage, Low-contrast objects, Regions with subtle texture changes, Hidden shapes in cluttered environments, Objects mimicking surroundings"
t3 = "Marine animals, Fish with distinctive patterns, Underwater creatures, Sea organisms with unique shapes, Coral and aquatic plants, Ocean wildlife, Fluorescent sea creatures"
t4 = "Shiny surfaces, Reflective regions, Bright mirrored areas, Objects with glossy finishes, High-gloss materials, Light-bouncing surfaces, Mirror-like reflections"
t5 = "Polyps in medical images, Irregular growths on tissue, Smooth or lobulated lesions, Abnormal structures in scans, Rounded or oval shapes, Bright or dark regions against normal tissue, Polyp outlines with soft edges, Contrast-enhanced abnormal regions"
# t6 = "Outdoor shadows under sunlight, Shadows created by artificial lights, Urban shadows cast by buildings or trees,Faint shadows blending into textured surfaces, Shadows overlapping with dark regions, Complex interactions between shadow edges and object boundaries"


def main(args):
    task_counts = {
        t1: 10553,
        t2: 4040,
        t3: 4283,
        t4: 8159,
        t5: 1450,
        # t6: 4085,
    }

    dataset = FullDataset(args.train_image_path, args.train_mask_path, 352, mode='train')
    #
    total_samples = sum(task_counts.values())
    task_weights = {k: total_samples / v for k, v in task_counts.items()}
    print(task_weights)

    # 根据任务标签为每个样本分配权重
    sample_weights = [task_weights[task] for task in dataset.text]  # 假设 dataset.text 对应任务标签
    # 创建 WeightedRandomSampler
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=10553*3, replacement=True)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler, num_workers=8)
    # dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)

    device = torch.device("cuda")
    model = SAM2UNet(args.hiera_path)
    model.to(device)

    experts_params = []
    other_params = []
    for name, param in model.named_parameters():
        if isinstance(param, nn.Parameter) and param.requires_grad:
            if "experts" in name:  # 针对 MoE_Adapter 中 gates 的参数
                experts_params.append(param)
            else:
                other_params.append(param)

    optim = opt.AdamW([
        {"params": experts_params, "lr": 5e-4},  # gates 的学习率
        {"params": other_params, "initia_lr": args.lr},  # 其他部分的学习率
    ],
                      lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optim, args.epoch, eta_min=1.0e-6)
    os.makedirs(args.save_path, exist_ok=True)
    for epoch in range(args.epoch):

        for i, batch in enumerate(dataloader):
            x = batch['image']

            target = batch['label']
            text= batch['text']

            x = x.to(device)
            target = target.to(device)
            optim.zero_grad()
            # pred0, pred1, pred2,t_loss,p_loss,n_loss= model(x,text)
            pred0, pred1, pred2= model(x,text)
            loss0 = structure_loss(pred0, target)
            loss1 = structure_loss(pred1, target)
            loss2 = structure_loss(pred2, target)
            loss = loss0 + loss1 + loss2
            # tloss = loss+con_loss+aux_loss
            loss.backward()
            optim.step()
            if i % 50 == 0:
                # print("epoch:{}-{}: loss:{},p_loss:{},n_loss:{}".format(epoch + 1, i + 1, loss.item(),p_loss.item(),n_loss.item()))
                print("epoch:{}-{}: loss:{}".format(epoch + 1, i + 1, loss.item()))

        scheduler.step()
        if (epoch+1) % 5 == 0 or (epoch+1) == args.epoch:
            torch.save(model.state_dict(), os.path.join(args.save_path, 'SAM2-UNet-%d.pth' % (epoch + 1)))
            print('[Saving Snapshot:]', os.path.join(args.save_path, 'SAM2-UNet-%d.pth'% (epoch + 1)))


def seed_torch(seed=1024):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    seed_torch(1024)
    main(args)