import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib.patches as patches
from torchvision.transforms.functional import crop, resize, five_crop
from torchvision.transforms import RandomResizedCrop
from PIL import Image
from perlin_numpy import generate_perlin_noise_2d as gpn
from perlin_numpy import generate_fractal_noise_2d as gfn
from skimage.draw import random_shapes as rs
import random

NML = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
TT  = transforms.ToTensor()
transform = transforms.Compose([TT, NML])
target_w = 128
target_h = 128
min_scale = 0.5

class AugDataset(Dataset):
    def __init__(self, dirname, transform=transform, overlap_chance=2/3):
        self.dirname = dirname
        self.datalist = sorted(os.listdir(dirname))
        self.transform = transform
        self.orig_w = 256
        self.orig_h = 256
        self.goal_w = target_w 
        self.goal_h = target_h
        self.overlap_chance = overlap_chance

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img0 = Image.open(os.path.join(self.dirname, self.datalist[idx])).convert('RGB')
        assert img0.size == (self.orig_w, self.orig_h)

        # sample non-overlap cases
        overlap = random.random() < self.overlap_chance # chance of overlap
        if overlap:
            # sample img1 center coordinate
            xmin = self.goal_w // 2
            xmax = self.orig_w - self.goal_w // 2
            ymin = self.goal_h // 2
            ymax = self.orig_h - self.goal_h // 2
            img1_x = random.randint(xmin, xmax)
            img1_y = random.randint(ymin, ymax)
            # calc img1 boundary coordinate
            img1_xmin = img1_x - self.goal_w//2
            img1_xmax = img1_x + self.goal_w//2
            img1_ymin = img1_y - self.goal_h//2
            img1_ymax = img1_y + self.goal_h//2 
            # sample img2 center coordinate
            inview = random.choice([True, False]) # sample second crop entirely within the bounds of the first crop
            if inview:
            # inview = False
                scaling = random.uniform(min_scale, 1) # img2 can be a smaller crop than img1
            else:
                scaling = 1.0
            img2_w = round(self.goal_w * scaling)
            img2_h = round(self.goal_h * scaling)
            if inview: 
                img2_x = random.randint(img1_xmin+img2_w//2, img1_xmax-img2_w//2)
                img2_y = random.randint(img1_ymin+img2_h//2, img1_ymax-img2_h//2)
            else:
                img2_x = random.randint(max(img1_xmin, xmin), min(img1_xmax, xmax))
                img2_y = random.randint(max(img1_ymin, ymin), min(img1_ymax, ymax))
            # calc img2 boundary cooridnate
            img2_xmin = img2_x - img2_w//2 
            img2_xmax = img2_x + img2_w//2 
            img2_ymin = img2_y - img2_h//2 
            img2_ymax = img2_y + img2_h//2 
            assert 0 <= img2_xmin < img2_xmax <= self.orig_w
            assert 0 <= img2_ymin < img2_ymax <= self.orig_h
            # cropping img1 and img2
            crop1 = crop(img0, img1_ymin, img1_xmin, self.goal_h, self.goal_w)
            crop2 = crop(img0, img2_ymin, img2_xmin, img2_h, img2_w)
            crop2 = resize(crop2, (self.goal_h, self.goal_w)) # resize crop2 back to the size of crop1
            # calc relative coordinate
            rltv_x = (img2_x - img1_xmin) / self.goal_w
            rltv_y = (img2_y - img1_ymin) / self.goal_h
            assert 0 <= rltv_x <= 1
            assert 0 <= rltv_y <= 1
            rltv_c = torch.tensor((rltv_y, rltv_x, scaling))
        else: # nonoverlap
            assert self.orig_h == 256
            assert self.goal_h == 128
            fivecrops = five_crop(img0, self.goal_h)
            c1idx, c2idx = random.sample([0, 1, 2, 3], 2) # sample 2 unique idx without repetition
            crop1 = fivecrops[c1idx]
            crop2 = fivecrops[c2idx]
            # crop2 = RandomResizedCrop((128, 128), scale=(0.5, 1.0), ratio=(1.0, 1.0))(crop2)
            rltv_c = torch.tensor((0.0, 0.0, 0.0))
            crop_idx_to_coord = {0: (64, 64),
                                 1: (64, 191),
                                 2: (191, 64), 
                                 3: (191, 191)}
            img1_y, img1_x = crop_idx_to_coord[c1idx]
            img2_y, img2_x = crop_idx_to_coord[c2idx]

        # data transform
        img0 = self.transform(img0)
        crop1 = self.transform(crop1)
        crop2 = self.transform(crop2)
        crops = torch.cat((crop1, crop2), dim=0)
        img1_c = torch.tensor((img1_y, img1_x))
        img2_c = torch.tensor((img2_y, img2_x)) 
        
        return img0, crops, img1_c, img2_c, rltv_c


def tensor2np(img):
    img = img.detach().numpy()
    img = img / 2 + 0.5
    img = np.transpose(img, (1, 2, 0))
    return img 


def setup_ax(fig, gridspec, c='k'):
    ax = fig.add_subplot(gridspec)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.setp(ax.spines.values(), color=c, linewidth=3)
    return ax


def draw_bbox(ax, coord, c='r', pixel_length=None, scaling=None, target_w=target_w, target_h=target_h):
    # coord of format (y, x)
    if pixel_length is not None:
        w = pixel_length
        h = pixel_length
    else:
        w = target_w
        h = target_h
    if scaling is not None:
        w = (w * scaling)#.round()
        h = (h * scaling)#.round()
    top = coord[0] - h//2
    left = coord[1] - w//2
    rect = patches.Rectangle((left, top), w, h, 
                            linewidth=3, edgecolor=c, facecolor='none')
    ax.add_patch(rect)


def plot(data, num, rc_pd=None, pixel=None):
    img0, imgs, c1, c2, rc = data
    img1 = imgs[:, :3]
    img2 = imgs[:, 3:]
    fig = plt.figure()
    # fig.subplots_adjust(0.01,0.01,0.99,0.99)
    plt.tight_layout()
    gs = gridspec.GridSpec(num, 3, hspace=0, wspace=0.05)
    for i in range(num):
        im0 = tensor2np(img0[i])
        ax = setup_ax(fig, gs[i, 0])
        ax.imshow(im0)
        draw_bbox(ax, c1[i], 'r')
        draw_bbox(ax, c2[i], 'g', scaling=rc[i, 2])
        ax.scatter(c2[i, 1], c2[i, 0], color='g')
        # setup crop1
        im1 = tensor2np(img1[i])
        ax = setup_ax(fig, gs[i, 1], c='r')
        ax.imshow(im1)
        x_gt = rc[i, 1]*target_w
        y_gt = rc[i, 0]*target_h
        # plot bounding box of crop2 in crop1
        ax.scatter(x_gt, y_gt, color='g')
        # draw_bbox(ax, (y_gt, x_gt), c='g', scaling=rc[i, 2])
        if rc_pd is not None:
            x_pd = rc_pd[i, 1]*target_w
            y_pd = rc_pd[i, 0]*target_h
            ax.scatter(x_pd, y_pd, color='b')
            ax.plot((x_pd, x_gt), (y_pd, y_gt), color='r')
        # ax.set_title([round(rc[i][0].item(), 2), round(rc[i][1].item(), 2), round(rc[i][2].item(), 2)], fontsize=8)
        # setup crop2
        im2 = tensor2np(img2[i])
        ax = setup_ax(fig, gs[i, 2], c='g')
        ax.imshow(im2)
        draw_bbox(ax, (64, 64), c='g', pixel_length=pixel)


def gen_pn(save_path, num=10000):
    for i in range(num):
        res = random.choice([2, 4, 8])
        clip = random.choice([True, False])
        n1 = gpn((256, 256), (res, res))
        n2 = gpn((256, 256), (res, res))
        n3 = gpn((256, 256), (res, res))
        n = np.stack((n1, n2, n3), axis=2)
        if clip:
            img = n.clip(0, 1) * 255
        else:
            img = (n - n.min()) / n.ptp() * 255
        img = Image.fromarray(img.astype('uint8')).convert('RGB')
        img.save(os.path.join(save_path, str(i).zfill(5) + '.png'))        


def gen_fn(save_path, num=10000):
    for i in range(num):
        res = random.choice([2, 4, 8])
        octave = random.choice([1, 2, 3, 4, 5])
        clip = random.choice([True, False])
        n1 = gfn((256, 256), (res, res), octave)
        n2 = gfn((256, 256), (res, res), octave)
        n3 = gfn((256, 256), (res, res), octave)
        n = np.stack((n1, n2, n3), axis=2)
        if clip:
            img = n.clip(0, 1) * 255
        else:
            img = (n - n.min()) / n.ptp() * 255
        img = Image.fromarray(img.astype('uint8')).convert('RGB')
        img.save(os.path.join(save_path, str(i).zfill(5) + '.png'))        


def gen_rand_shape(save_path, num=10000, allow_overlap=True):
    for i in range(num):
        if allow_overlap:
            img, _ = rs((256, 256), min_shapes=80, max_shapes=100, min_size=20, max_size=50,
                intensity_range=((0, 255),), allow_overlap=allow_overlap)
        else:
            img, _ = rs((256, 256), min_shapes=60, max_shapes=80, min_size=10, max_size=30,
                intensity_range=((0, 255),), allow_overlap=allow_overlap)
        img = Image.fromarray(img.astype('uint8')).convert('RGB')
        img.save(os.path.join(save_path, str(i).zfill(5) + '.png'))


def gen_noise(save_path, num=10000):
    for i in range(num):
        img = np.random.rand(256, 256, 3) * 255
        img = Image.fromarray(img.astype('uint8')).convert('RGB')
        img.save(os.path.join(save_path, str(i).zfill(5) + '.png'))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', type=str, required=True)
    parser.add_argument('--num', type=int, default=10000)
    parser.add_argument('--save_path', required=True)
    parser.add_argument('--overlap', action='store_true')

    args = parser.parse_args()
    assert args.type in ['fractal', 'perlin', 'shape']
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    if args.type == 'fractal':
        gen_fn(args.save_path, args.num)
    elif args.type == 'perlin':
        gen_pn(args.save_path, args.num)
    elif args.type == 'shape':
        gen_rand_shape(args.save_path, args.num, allow_overlap=args.overlap)

