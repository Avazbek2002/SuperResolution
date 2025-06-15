import glob
import os


if __name__ == '__main__':
    folder_path = r"D:\Fargona_clips"
    tif_files = glob.glob(os.path.join(folder_path, '**', '*.tif'), recursive=True)
    for tif_file in tif_files:
        print(f"Processing file: {tif_file}")