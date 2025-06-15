import os
import random
import tifffile as tiff
from torch.utils.data import Dataset
from PIL import TiffImagePlugin
import util as Util
import sys
import numpy as np
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import core.metrics as Metrics
import cv2
from torch.utils.data import DataLoader

TiffImagePlugin.OPEN = True

class SuperResolutionDataset(Dataset):
    """
    A custom dataset that:
      1. Expects a list of (image_path, metadata_dict) tuples
      2. Loads high-resolution (HR) images from disk
      3. Applies a random downsampling operation to produce low-resolution (LR) images
      4. Returns (LR_image, HR_image, metadata_dict)
    """
    def __init__(self, image_metadata_list, downsampling_factor=4, transforms=None):
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
        self.downsample_methods = [cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_LANCZOS4]

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


import matplotlib.pyplot as plt

def visualize_get_item(dataset, idx):
    """
    Visualizes the HR, LR, and SR images returned by dataset.__getitem__(idx).
    
    Args:
        dataset: An instance of the dataset containing __getitem__ method.
        idx: Index of the image to visualize.
    """
    # Get the image dictionary
    data = dataset[idx]
    
    # Convert tensors to numpy images
    hr_img = Metrics.tensor2img(data['HR'], out_type=np.uint16, min_max=(-1, 1))
    lr_img = Metrics.tensor2img(data['LR'], out_type=np.uint16, min_max=(-1, 1))

    Metrics.save_img(hr_img, 'test_hr.png')
    Metrics.save_img(lr_img, 'test_lr.png')

    # Create a plot
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    axs[0].imshow(lr_img)
    axs[0].set_title('Low-Resolution (LR)')
    axs[0].axis('off')

    axs[1].imshow(hr_img)
    axs[1].set_title('High-Resolution (HR)')
    axs[1].axis('off')

    plt.tight_layout()
    plt.show()

fargona_dir = r"D:\SuperResolutionProject\Fargona\ReadyData"
samarkand_dir = r"D:\SuperResolutionProject\Samarkand\ReadyData"

# 2. Gather (image_path, metadata_dict) for all .tiff images
combined_data = gather_images(
    root_dirs=[fargona_dir, samarkand_dir],
    extension=".tif"   # or ".tif" if needed
)

# 3. Create your PyTorch dataset
full_dataset = SuperResolutionDataset(
    image_metadata_list=combined_data,
    downsampling_factor=4   # adjust as appropriate
)

downsampling_factor = 4  # Factor by which HR images will be downsampled
batch_size = 16          # Number of samples per batch
num_workers = 4          # Number of worker threads for data loading
shuffle = True  

dataloader = DataLoader(
    full_dataset,
    batch_size=batch_size,
    shuffle=shuffle,
    num_workers=num_workers,
    pin_memory=True  # Recommended if using a GPU
)

def plot_tensor_histogram(tensor, title="Tensor Value Distribution", bins=50):
    # Convert the tensor to a NumPy array
    tensor_np = tensor.cpu().numpy()
    
    # Flatten the tensor to 1D
    tensor_flat = tensor_np.flatten()
    
    # Plot the histogram
    plt.figure(figsize=(8, 6))
    plt.hist(tensor_flat, bins=bins, color='blue', alpha=0.7)
    plt.title(title)
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    for i, batch in enumerate(dataloader):  # Use enumerate to track the iteration index

        # Print min-max range every 1000 iterations
        if i % 100 == 0:
            lr_images = batch['LR']  # Low-resolution images
            hr_images = batch['HR']  # High-resolution images
            sr_images = batch['SR']  # Super-resolution images (optional)
            indices = batch['Index']  # Indices of the images
            lr_min, lr_max = lr_images.min().item(), lr_images.max().item()
            hr_min, hr_max = hr_images.min().item(), hr_images.max().item()
            plot_tensor_histogram(lr_images, title=f"LR Image Histogram - Iteration {i}")
            plot_tensor_histogram(hr_images, title=f"HR Image Histogram - Iteration {i}")
            print(f"Iteration {i}:")
            print(f"  LR Image - Min: {lr_min}, Max: {lr_max}")
            print(f"  HR Image - Min: {hr_min}, Max: {hr_max}")

