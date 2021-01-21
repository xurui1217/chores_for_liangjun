import os
import logging
if __name__ == '__main__':
    for mode in ['train', 'test']:
        print(f'mode:{mode}')
        file = open(mode+'.txt', 'w')
        if mode == 'train':
            # allimg = os.listdir('/home/nip/LabDatasets/pixelwise/WP6/images')
            allimg = os.listdir('/home/xr/sjtu/xr/JAS/pixelwise/WP1/images')
        else:
            # allimg = os.listdir('/home/nip/LabDatasets/pixelwise/WP6/test/images')
            allimg = os.listdir(
                '/home/xr/sjtu/xr/JAS/pixelwise/WP1/test/images')
        for i in range(len(allimg)):
            print(f'process No:{i},allimg={allimg[i]}')
            name = allimg[i] + '\n' if i < len(allimg) - 1 else allimg[i]
            file.writelines(name)
        file.close()
