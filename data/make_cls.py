import pickle
import os
from PIL import Image
import numpy as np
from tqdm import tqdm
import torch
# root = '/home/nip/LabDatasets/pixelwise/WP1' #type the corresponding file path
root = '/home/xr/sjtu/xr/JAS/pixelwise/WP1'


def make_one_hot(input, num_classes):
    """Convert class index tensor to one hot encoding tensor.
    Args:
         input: A tensor of shape [N, 1, *]
         num_classes: An int of number of class
    Returns:
        A tensor of shape [N, num_classes, *]
    """
    shape = np.array(input.shape)
    shape[1] = num_classes
    shape = tuple(shape)
    result = torch.zeros(shape)
    result = result.scatter_(1, input.cpu(), 1)

    return result


if __name__ == '__main__':

    for mode in ['train', 'test']:
        if mode == 'test':
            mask_path = os.path.join(root, mode, 'classes')
            save_path = os.path.join(root, 'cls', mode)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            data_list = [l.strip('\n') for l in open(os.path.join(
                root,  'test.txt')).readlines()]
        else:
            mask_path = os.path.join(root, 'classes')
            save_path = os.path.join(root, 'cls', mode)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            data_list = [l.strip('\n') for l in open(os.path.join(
                root,  'train.txt')).readlines()]
        item = []
        class_num = 0
        dic = {}
        for ind, it in enumerate(data_list):
            mask = Image.open(os.path.join(mask_path, it))
            mask = np.array(mask)
            if ind < 1:
                for i in range(mask.shape[0]):
                    for j in range(mask.shape[1]):
                        if tuple(mask[i, j, :]) not in dic:
                            dic[tuple(mask[i, j, :])] = class_num
                            class_num += 1
            else:
                break
        if mode == 'train':
            save_pallete = os.path.join(root, 'pallete.txt')
            p2 = open(save_pallete, 'wb')
            pickle.dump(dic, p2)
            p2.close()
        for it in tqdm(data_list):
            mask = Image.open(os.path.join(mask_path, it))
            mask = np.array(mask)
            cls = np.zeros((mask.shape[0], mask.shape[1]), dtype=int)

            for i in range(mask.shape[0]):
                for j in range(mask.shape[1]):
                    attri = dic[tuple(mask[i, j, :])]
                    cls[i, j] = attri
            np.save(os.path.join(save_path, it[:-4]+'.npy'), cls)
