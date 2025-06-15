import os
import torch
import torchvision
import random
import numpy as np
import rasterio
from rasterio.windows import Window
import numpy as np 
from rasterio.errors import RasterioIOError
import glob

IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG',
                  '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tif', '.TIF']


def is_image_file(filename):
    return filename.endswith('.tif') or filename.endswith('.TIF')


def get_paths_from_images(path):
    assert os.path.isdir(path), '{:s} is not a valid directory'.format(path)
    images = []
    for dirpath, _, fnames in sorted(os.walk(path)):
        for fname in sorted(fnames):
            if is_image_file(fname):
                img_path = os.path.join(dirpath, fname)
                images.append(img_path)
    assert images, '{:s} has no valid image file'.format(path)
    return sorted(images)


def augment(img_list, hflip=True, rot=True, split='val'):
    # horizontal flip OR rotate
    hflip = hflip and (split == 'train' and random.random() < 0.5)
    vflip = rot and (split == 'train' and random.random() < 0.5)
    rot90 = rot and (split == 'train' and random.random() < 0.5)

    def _augment(img):
        if hflip:
            img = img[:, ::-1, :]
        if vflip:
            img = img[::-1, :, :]
        if rot90:
            img = img.transpose(1, 0, 2)
        return img

    return [_augment(img) for img in img_list]


# def transform2numpy(img):
#     img = np.array(img)
#     img = img.astype(np.float32) / 65535.
#     if img.ndim == 2:
#         img = np.expand_dims(img, axis=2)
#     # some images have 4 channels
#     if img.shape[2] > 3:
#         img = img[:, :, :3]
#     return img

def transform2numpy(img):
    img = np.array(img).astype(np.float32)
    
    # Apply min-max normalization to scale values to [0, 1]
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
    # Ensure the image has at most 3 channels
    if img.shape[2] > 3:
        img = img[:, :, :3]
    return img


def transform2tensor(img, min_max=(0, 1)):
    # HWC to CHW
    img = torch.from_numpy(np.ascontiguousarray(
        np.transpose(img, (2, 0, 1)))).float()
    # to range min_max
    img = img*(min_max[1] - min_max[0]) + min_max[0]
    return img


# implementation by numpy and torch
def transform_augment(img_list, split='val', min_max=(0, 1)):
    imgs = [transform2numpy(img) for img in img_list]
    imgs = augment(imgs, split=split)
    ret_img = [transform2tensor(img, min_max) for img in imgs]
    return ret_img


# implementation by torchvision, detail in https://github.com/Janspiry/Image-Super-Resolution-via-Iterative-Refinement/issues/14
# totensor = torchvision.transforms.ToTensor()
# hflip = torchvision.transforms.RandomHorizontalFlip()
# def transform_augment(img_list, split='val', min_max=(0, 1)):    
#     imgs = [totensor(img) for img in img_list]
#     if split == 'train':
#         imgs = torch.stack(imgs, 0)
#         imgs = hflip(imgs)
#         imgs = torch.unbind(imgs, dim=0)
#     ret_img = [img * (min_max[1] - min_max[0]) + min_max[0] for img in imgs]
#     return ret_img

# def transform_augment(img_list, split='val', min_max=(0, 1)):
#     do_flip = (split == 'train') and (random.random() > 0.5)

#     ret_img = []
#     for img in img_list:
#         if do_flip:
#             img = TF.hflip(img)
#         tensor = totensor(img)
#         tensor = tensor * (min_max[1] - min_max[0]) + min_max[0]
#         ret_img.append(tensor)
#     return ret_img

def tile_raster_and_filter_black(input_tiff, output_dir, tile_size=(512, 512), black_threshold_percent=30, random_word=None):
    """
    Divides a large TIFF raster into smaller tiles, replaces nodata
    values with 0, and filters out tiles that are predominantly black.

    Args:
        input_tiff (str): Path to the large input TIFF file.
        output_dir (str): Directory to save the output tiles.
        tile_size (tuple): A tuple of (width, height) for the tiles.
        black_threshold_percent (float): Percentage of black area above which
                                         a tile will be discarded.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    skipped_patches_count = 0
    saved_patches_count = 0

    try:
        with rasterio.open(input_tiff) as src:
            width = src.width
            height = src.height
            profile = src.profile
            nodata_value = src.nodata
            
            tile_w, tile_h = tile_size

            print(f"Starting tiling process. Input: {input_tiff}")
            print(f"Nodata value in source: {nodata_value}")
            print(f"Tiles with more than {black_threshold_percent}% black area will be skipped.")

            for i in range(0, height, tile_h):
                for j in range(0, width, tile_w):
                    # Define the actual window width and height, handling edges
                    window_w = min(tile_w, width - j)
                    window_h = min(tile_h, height - i)
                    
                    window = Window(j, i, window_w, window_h)

                    try:
                        # Read the data from the window
                        data = src.read(window=window)
                    except RasterioIOError as e:
                        print(f"Error reading window at ({j},{i}) in {input_tiff}: {e}")
                        print(f"Skipping this window and continuing with the next.")
                        skipped_patches_count += 1 # Count skipped due to read error
                        continue # Skip to the next window

                    # Handle nodata values by replacing them with 0
                    if nodata_value is not None:
                        # Ensure we operate on a copy if modification is needed
                        if np.any(data == nodata_value): # Check if nodata_value actually exists in this patch
                            data = data.copy() 
                            data[data == nodata_value] = 0
                        
                    # Calculate the percentage of black pixels
                    # A pixel is black if all its band values are 0
                    if data.ndim == 3: # Multi-band image (bands, height, width)
                        # Check for all zeros across the band axis (axis=0)
                        is_black_pixel = np.all(data == 0, axis=0)
                    elif data.ndim == 2: # Single-band image (height, width)
                        is_black_pixel = (data == 0)
                    else: # Should not happen for typical rasters
                        print(f"Warning: Patch at ({i},{j}) has unexpected data dimensions: {data.shape}. Skipping analysis for this patch.")
                        skipped_patches_count += 1
                        continue

                    black_pixel_count = np.sum(is_black_pixel)
                    total_pixels_in_patch = data.shape[-2] * data.shape[-1] # height * width of the actual patch

                    if total_pixels_in_patch == 0: # Should not happen with valid window_w/h
                        skipped_patches_count += 1
                        print(f"Skipping empty patch at ({i},{j}).")
                        continue
                        
                    black_area_percentage = (black_pixel_count / total_pixels_in_patch) * 100

                    # If black area is below or equal to the threshold, save the patch
                    if black_area_percentage <= black_threshold_percent:
                        transform = src.window_transform(window)
                        
                        tile_profile = profile.copy()
                        tile_profile.update({
                            'height': window_h, # Use actual window height
                            'width': window_w,  # Use actual window width
                            'transform': transform,
                            'nodata': None # Nodata has been replaced with 0
                        })

                        output_filename = os.path.join(output_dir, f'{random_word}_tile_{i}_{j}_v4.tif')
                        try:
                            with rasterio.open(output_filename, 'w', **tile_profile) as dst:
                                dst.write(data)
                            saved_patches_count += 1
                        except Exception as e: # Catch potential errors during writing as well
                            print(f"Error writing tile {output_filename}: {e}")
                            skipped_patches_count += 1
                    else:
                        skipped_patches_count += 1
                        
            print(f"\nTiling complete for {input_tiff}.")
            print(f"Saved {saved_patches_count} patches in '{output_dir}'.")
            print(f"Skipped {skipped_patches_count} patches (due to black area or read/write errors).")
            
            # This 'os.remove' will only execute if rasterio.open(input_tiff) was successful
            os.remove(input_tiff) 
            print(f"Original file '{input_tiff}' removed.")

    except RasterioIOError as e:
        print(f"Error opening or reading file {input_tiff}: {e}")
        print(f"Deleting corrupted file: {input_tiff}")
        try:
            os.remove(input_tiff)
            print(f"File '{input_tiff}' successfully deleted.")
        except OSError as oe:
            print(f"Error deleting file '{input_tiff}': {oe}")
    except Exception as e: # Catch any other unexpected errors during file processing
        print(f"An unexpected error occurred while processing {input_tiff}: {e}")
        print(f"Attempting to delete file '{input_tiff}' due to unexpected error.")
        try:
            os.remove(input_tiff)
            print(f"File '{input_tiff}' successfully deleted.")
        except OSError as oe:
            print(f"Error deleting file '{input_tiff}': {oe}")

def delete_non_tif_files(folder_path):
    """
    Deletes all files from a folder that are not .tif or .tiff files.
    If a file is inaccessible, it skips that file and moves to the next.

    Args:
        folder_path (str): The absolute path to the folder.
    """
    
    all_files = glob.glob(os.path.join(folder_path, '**', '*'), recursive=True)

    try:
        for file_path in all_files:            
            if os.path.isfile(file_path):
                if not file_path.lower().endswith('.tif'):
                    try:
                        print(f"Attempting to delete: {file_path}")
                        os.remove(file_path)
                        print(f"Successfully deleted: {file_path}")
                    except OSError as e:
                        print(f"Could not delete '{file_path}': {e}. Skipping this file.")
                else:
                    print(f"Keeping: {file_path} (is a .tif or .tiff file)")
    except FileNotFoundError:
        print(f"Error: The folder '{folder_path}' was not found.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")