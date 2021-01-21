import os
import numpy as np
# import scipy.io as sio
import torch
from PIL import Image
from torch.utils import data
import pickle
# root = '/home/nip/LabDatasets/pixelwise/WP1'
root = '/home/xr/sjtu/xr/JAS/pixelwise/WP1'
save_pallete = os.path.join(root, 'pallete.txt')
p2 = open(save_pallete, 'rb')
palete = pickle.load(p2)
p2.close()
# import torchvision.transforms as standard_transforms
# import sys
# sys.path.append('/home/nip/zf/chores')
# import utils.transforms as extended_transforms
# pale = {(0, 0, 0) (139847156440016): 0
# (0, 130, 255) (139847156484928):  1
# (125, 125, 125) (139847156510792) =  2
# (0, 255, 255) (139847180734680) =  3
# (255, 0, 255) (139847156511368) = 4
# (0, 0, 130) (139847156511440) =  5
# (255, 255, 255) (139847156511512) =  6
# (255, 0, 0) (139847156511584) =  7
# (0, 130, 0) (139847156511656) =  8
# (255, 255, 0) (139847156511728) =  9
# (130, 130, 0) (139847156511800) =  10
# (0, 255, 0) (139847156511872) =  11
# (0, 0, 255) (139847156511944) =  12}
num_classes = len(palete)
# ignore_label = 255


'''
'''
palette = [0, 0, 0,
           0, 130, 255,
           125, 125, 125,
           0, 255, 255,
           255, 0, 255,
           0, 0, 130,
           255, 255, 255,
           255, 0, 0,
           0, 130, 0,
           255, 255, 0,
           130, 130, 0,
           0, 255, 0,
           0, 0, 255]

zero_pad = 256 * 3 - len(palette)
for i in range(zero_pad):
    palette.append(0)


def colorize_mask(mask):
    # mask: numpy array of the mask
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)

    return new_mask


def make_dataset(mode):
    assert mode in ['train', 'val', 'test']
    items = []
    if mode in ['train']:
        img_path = os.path.join(root, 'images')
        mask_path = os.path.join(root, 'cls', mode)
        data_list = [l.strip('\n') for l in open(os.path.join(
            root, 'train.txt')).readlines()]
        for it in data_list:
            item = (os.path.join(img_path, it), os.path.join(mask_path, it))
            items.append(item)
    elif mode == 'val':
        img_path = os.path.join(root, 'test', 'images')
        mask_path = os.path.join(root, 'cls', 'test')
        data_list = [l.strip('\n') for l in open(os.path.join(
            root,  'test.txt')).readlines()]
        for it in data_list:
            item = (os.path.join(img_path, it), os.path.join(mask_path, it))
            items.append(item)
    else:
        img_path = os.path.join(root, 'test')
        data_list = [l.strip('\n') for l in open(os.path.join(
            root, 'test.txt')).readlines()]
        for it in data_list:
            items.append((img_path, it))
    return items


class Wp(data.Dataset):
    def __init__(self, mode, joint_transform=None, sliding_crop=None, transform=None, target_transform=None):
        self.imgs = make_dataset(mode)
        if len(self.imgs) == 0:
            raise RuntimeError('Found 0 images, please check the data set')
        self.mode = mode
        self.joint_transform = joint_transform
        self.sliding_crop = sliding_crop
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        if self.mode == 'test':
            img_path, img_name = self.imgs[index]
            img = Image.open(os.path.join(
                img_path, img_name + '.jpg')).convert('RGB')
            if self.transform is not None:
                img = self.transform(img)
            return img_name, img

        img_path, mask_path = self.imgs[index]
        img = Image.open(img_path).convert('RGB')
        if self.mode == 'train':
            mask = np.load(mask_path[:-3] + 'npy')
        else:
            mask = np.load(mask_path[:-3] + 'npy')

        if self.joint_transform is not None:
            img, mask = self.joint_transform(img, mask)

        if self.sliding_crop is not None:
            img_slices, mask_slices, slices_info = self.sliding_crop(img, mask)
            if self.transform is not None:
                img_slices = [self.transform(e) for e in img_slices]
            if self.target_transform is not None:
                mask_slices = [self.target_transform(e) for e in mask_slices]
            img, mask = torch.stack(img_slices, 0), torch.stack(mask_slices, 0)
            return img, mask, torch.LongTensor(slices_info)
        else:
            if self.transform is not None:
                img = self.transform(img)
            if self.target_transform is not None:
                mask = self.target_transform(mask)
            return img, mask

    def __len__(self):
        return len(self.imgs)


# if __name__ == '__main__':
#     mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#     input_transform = standard_transforms.Compose([
#         standard_transforms.ToTensor(),
#         standard_transforms.Normalize(*mean_std)
#     ])
#     target_transform = extended_transforms.MaskToTensor()
#     train_set = Wb('train', transform=input_transform, target_transform=target_transform)
#     z,b = train_set[0]
