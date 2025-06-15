import os
import random
import tifffile as tiff
from torch.utils.data import Dataset
from PIL import TiffImagePlugin
import data.util as Util
import cv2

TiffImagePlugin.OPEN = True

class SuperResolutionDataset(Dataset):
    """
    A custom dataset that:
      1. Expects a list of (image_path, metadata_dict) tuples
      2. Loads high-resolution (HR) images from disk
      3. Applies a random downsampling operation to produce low-resolution (LR) images
      4. Returns (LR_image, HR_image, metadata_dict)
    """
    def __init__(self, image_metadata_list, downsampling_factor=2, transforms=None):
        """
        Args:
            image_metadata_list: List of tuples (img_path, metadata_dict).
            downsampling_factor: Factor by which the images will be downsampled.
            transforms: Optional transforms (like ToTensor) applied after downsampling.
        """
        self.image_metadata_list = image_metadata_list
        self.downsampling_factor = downsampling_factor
        self.transforms = transforms
        # Different interpolation methods in Pillow:
        #   Image.NEAREST, Image.BILINEAR, Image.BICUBIC, Image.LANCZOS, etc.
        self.downsample_methods = [cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_LANCZOS4, cv2.INTER_AREA]

    def __len__(self):
        return len(self.image_metadata_list)

    def __getitem__(self, idx):
        img_path = self.image_metadata_list[idx]

        # Load the high-resolution image
        hr_image = tiff.imread(img_path)
        # hr_image = preprocess_tiff(img_array)

        # Randomly pick a downsampling method
        method = random.choice(self.downsample_methods)

        # Compute new size
        w, h, _ = hr_image.shape
        new_size = (w // self.downsampling_factor, h // self.downsampling_factor)
        # Create low-resolution image using the chosen interpolation
        lr_image = cv2.resize(hr_image, new_size, interpolation=method)

        [lr_image, hr_image] = Util.transform_augment(
                [lr_image, hr_image], split='train', min_max=(-1, 1))

        return  {'HR': hr_image, 'LR': lr_image, 'SR': hr_image ,'Index': idx}

def gather_images(root_dirs, extension="tif"):
    """
    Gathers all images with the given extension from the list of `root_dirs`,
    along with their metadata from (for example) a CSV in each subfolder.
    
    Returns:
        A list of tuples: [(img_path, metadata_dict), ...].
    """
    image_metadata_list = []

    for root_dir in root_dirs:
        # Go through each subfolder inside root_dir
        for subfolder_name in os.listdir(root_dir):
            subfolder_path = os.path.join(root_dir, f"{subfolder_name}/images")

            for img_path in os.listdir(subfolder_path):
                if img_path.lower().endswith(extension):
                    image_metadata_list.append(os.path.join(subfolder_path, img_path))

    return image_metadata_list