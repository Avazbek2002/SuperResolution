import os
import shutil

# Define the source folders and the target folder
folder_names = ['D:\\SuperResolutionProject\\Fargona\\ReadyData', 'D:\\SuperResolutionProject\\Samarkand\\ReadyData']
target_folder = 'D:\\SuperResolutionProject\\CombinedImages'

# Create the target folder if it doesn't exist
os.makedirs(target_folder, exist_ok=True)

# Iterate through all subfolders and move `.tiff` images
for folder_name in folder_names:
    for subfolder_name in os.listdir(folder_name):
        subfolder_path = os.path.join(folder_name, subfolder_name)
        print(f"Processing folder: {subfolder_path}")

        # Check if the subfolder contains 'images' directory
        images_dir = os.path.join(subfolder_path, 'images')
        if os.path.exists(images_dir):
            print(f"Found images directory: {images_dir}")
            # List all `.tiff` files in the images directory
            for img_file in os.listdir(images_dir):
                if img_file.endswith('.tif'):
                    source_path = os.path.join(images_dir, img_file)
                    target_path = os.path.join(target_folder, img_file)
                    print(f"Moving {source_path} to {target_path}")
                    shutil.move(source_path, target_path)
        else:
            print(f"No images directory found in {subfolder_path}")