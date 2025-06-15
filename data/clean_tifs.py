import os
import rasterio
from rasterio.errors import RasterioIOError
import concurrent.futures
import threading
import time

# --- Configuration ---
# The number of parallel threads to use for processing files.
NUM_WORKERS = 16
# The target dimensions for the images.
TARGET_WIDTH = 512
TARGET_HEIGHT = 512

def process_file(file_path, print_lock):
    """
    Worker function to process a single image file.
    This function is executed by each thread.
    
    Args:
        file_path (str): The full path to the image file to process.
        print_lock (threading.Lock): A lock to synchronize print statements.
    """
    try:
        # Use a 'with' statement to ensure the dataset is properly closed
        with rasterio.open(file_path) as dataset:
            width = dataset.width
            height = dataset.height

            if width == TARGET_WIDTH and height == TARGET_HEIGHT:
                status_message = f"KEEPING (size: {width}x{height})"
                # Return False to indicate the file was not deleted
                return status_message, False
            else:
                status_message = f"DELETING (size: {width}x{height})"
                # Return True to indicate the file should be deleted
                return status_message, True

    except RasterioIOError:
        status_message = "SKIPPING - Cannot read raster file. It may be corrupted."
        return status_message, False
    except Exception as e:
        status_message = f"SKIPPING - An unexpected error occurred: {e}"
        return status_message, False

def clean_rasters_multithreaded(folder_path):
    """
    Scans a folder and deletes TIFF images that do not match the target dimensions
    using a pool of worker threads.
    """
    print(f"Starting scan of '{folder_path}' with {NUM_WORKERS} worker threads...\n")
    start_time = time.time()
    
    # Create a thread-safe lock for printing to the console
    print_lock = threading.Lock()
    
    # Step 1: Get a list of all files to process
    try:
        files_to_process = [
            os.path.join(folder_path, f)
            for f in os.listdir(folder_path)
            if f.lower().endswith(('.tif', '.tiff'))
        ]
    except FileNotFoundError:
        print(f"Error: The directory '{folder_path}' was not found.")
        return

    if not files_to_process:
        print("No .tif or .tiff files found in this directory.")
        return

    files_to_delete = []
    
    # Step 2: Use ThreadPoolExecutor to process files in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        # Submit each file to be processed by the worker function
        future_to_file = {
            executor.submit(process_file, file_path, print_lock): file_path
            for file_path in files_to_process
        }
        
        for i, future in enumerate(concurrent.futures.as_completed(future_to_file)):
            file_path = future_to_file[future]
            filename = os.path.basename(file_path)
            try:
                message, should_delete = future.result()
                if should_delete:
                    files_to_delete.append(file_path)
                
                # Use the lock to print safely from the main thread
                with print_lock:
                    print(f"[{i+1}/{len(files_to_process)}] {filename}: {message}")

            except Exception as e:
                with print_lock:
                    print(f"Error processing {filename}: {e}")

    # Step 3: Perform the actual file deletion in the main thread
    if files_to_delete:
        print("\n--- Deleting Files ---")
        for file_path in files_to_delete:
            try:
                os.remove(file_path)
                print(f"Successfully deleted {os.path.basename(file_path)}")
            except OSError as e:
                print(f"Error deleting {os.path.basename(file_path)}: {e}")
        print("--------------------")
    else:
        print("\nNo files needed to be deleted.")

    end_time = time.time()
    print(f"\nScan complete. Total time: {end_time - start_time:.2f} seconds.")


if __name__ == "__main__":
    # The script will check the folder it is currently in.
    # To specify a different folder, change the path below.
    # For example: folder_to_scan = "C:\\Users\\a.isroilov\\Desktop\\patches"
    folder_to_scan = r"C:\Users\a.isroilov\Desktop\patches"

    clean_rasters_multithreaded(folder_to_scan)