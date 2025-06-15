from util import tile_raster_and_filter_black
from util import delete_non_tif_files
import nltk
from nltk.corpus import words
import argparse
import random
import os
import glob

def patchify_tiff(input_folder, output_directory, patch_size=(512, 512), black_area_threshold=70):
    """
    Patchify a TIFF file into smaller patches, filtering out patches with too much black area.
    
    :param input_tiff: Path to the input TIFF file.
    :param output_directory: Directory where the patches will be saved.
    :param patch_size: Size of the patches (width, height).
    :param black_threshold_percent: Percentage of black pixels allowed in a patch.
    :param random_word: Optional random word to use in the filename.
    """
    tif_files = glob.glob(os.path.join(input_folder, '**', '*.tif'), recursive=True)
    nltk.download('words')  # Ensure nltk words corpus is downloaded
    words_list = words.words()  # Load the list of words
    random_word = random.choice(words_list) + "_" + random.choice(words_list)  # Randomly select a word
    print(f"Using random word for filename: {random_word}")

    # Ensure the output directory exists
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    for input_tiff in tif_files:

        tile_raster_and_filter_black(
            input_tiff,
            output_directory,
            tile_size=patch_size,
            black_threshold_percent=black_area_threshold,
            random_word=random_word
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process TIFF images by deleting non-TIFF files and creating patches.")

    parser.add_argument('input_folder', type=str,
                        help='Path to the input folder containing TIFF files.')
    parser.add_argument('output_directory', type=str,
                        help='Path to the output directory where patches will be saved.')
    
    args = parser.parse_args()

    input_folder = args.input_folder
    output_directory = args.output_directory

    delete_non_tif_files(input_folder)

    patchify_tiff(input_folder, output_directory)