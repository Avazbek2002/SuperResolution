from io import BytesIO
from PIL import Image
from torch.utils.data import Dataset
import random
import data.util as Util
import tifffile as tiff
from PIL import TiffImagePlugin
import cv2

TiffImagePlugin.OPEN = True


class LRHRDataset(Dataset):
    def __init__(self, dataroot, datatype, l_resolution=16, r_resolution=128, split='train', data_len=-1, need_LR=False):
        self.datatype = datatype
        self.l_res = l_resolution
        self.r_res = r_resolution
        self.data_len = data_len
        self.need_LR = need_LR
        self.split = split

        self.sr_path = Util.get_paths_from_images(
            '{}/sr_{}_{}'.format(dataroot, l_resolution, r_resolution))
        self.hr_path = Util.get_paths_from_images(
            '{}/hr_{}'.format(dataroot, r_resolution))
        if self.need_LR:
            self.lr_path = Util.get_paths_from_images(
                '{}/lr_{}'.format(dataroot, l_resolution))
        self.dataset_len = len(self.hr_path)
        if self.data_len <= 0:
            self.data_len = self.dataset_len
        else:
            self.data_len = min(self.data_len, self.dataset_len)

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        img_HR = None
        img_LR = None

        img_HR = tiff.imread(self.hr_path[index])
        img_SR = tiff.imread(self.sr_path[index])
        if self.need_LR:
            img_LR = tiff.imread(self.lr_path[index])
        if self.need_LR:
            [img_LR, img_SR, img_HR] = Util.transform_augment(
                [img_LR, img_SR, img_HR], split=self.split, min_max=(-1, 1))
            return {'LR': img_LR, 'HR': img_HR, 'SR': img_SR, 'Index': index}
        else:
            [img_SR, img_HR] = Util.transform_augment(
                [img_SR, img_HR], split=self.split, min_max=(-1, 1))
            return {'HR': img_HR, 'SR': img_SR, 'Index': index}
