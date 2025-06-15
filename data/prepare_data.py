import argparse
from multiprocessing import Lock, Process, RawValue
from functools import partial
from multiprocessing.sharedctypes import RawValue
from PIL import Image
from tqdm import tqdm
import os
from pathlib import Path
import numpy as np
import time
import tifffile as tiff
from PIL import TiffImagePlugin
import cv2

TiffImagePlugin.OPEN = True


def resize_and_convert(img, size, resample):
    if(img.shape[0] != size or img.shape[1] != size):
        img = cv2.resize(img, (size,size), interpolation=resample)
    return img


def resize_multiple(img, sizes=(256, 512), resample=Image.BICUBIC):
    lr_img = resize_and_convert(img, sizes[0], resample)
    hr_img = resize_and_convert(lr_img, sizes[1], resample) # This should be img, not lr_img
    sr_img = resize_and_convert(lr_img, sizes[1], resample)

    return [lr_img, hr_img, sr_img]

def resize_worker(img_file, sizes, resample, lmdb_save=False):
    try:
        img = tiff.imread(img_file)
        if img is None: # Check if tiff.imread returned None
            print(f"Warning: tifffile.imread returned None for {img_file}. Skipping.")
            return None, None
        
        # Add a check for empty or invalid image data (e.g., if it's not a proper numpy array)
        if not isinstance(img, np.ndarray) or img.size == 0:
            print(f"Warning: Image data is invalid or empty for {img_file}. Skipping.")
            return None, None

        out = resize_multiple(
            img, sizes=sizes, resample=resample)

        return img_file.name.split('.')[0], out
    except Exception as e:
        print(f"Error processing image {img_file}: {e}. Skipping.")
        return None, None # Return None to indicate failure


class WorkingContext():
    def __init__(self, resize_fn, lmdb_save, out_path, env, sizes):
        self.resize_fn = resize_fn
        self.lmdb_save = lmdb_save
        self.out_path = out_path
        self.env = env
        self.sizes = sizes

        self.counter = RawValue('i', 0)
        self.counter_lock = Lock()

    def inc_get(self):
        with self.counter_lock:
            self.counter.value += 1
            return self.counter.value

    def value(self):
        with self.counter_lock:
            return self.counter.value

def prepare_process_worker(wctx, file_subset):
    for file in file_subset:
        i, imgs = wctx.resize_fn(file)
        
        # If resize_fn failed, i or imgs will be None, so skip this file
        if i is None or imgs is None:
            continue

        lr_img, hr_img, sr_img = imgs
        
        try:
            # tiff.imwrite('{}/lr_{}/{}.tif'.format(wctx.out_path, wctx.sizes[0], i.zfill(5)), lr_img)
            tiff.imwrite('{}/hr_{}/{}.tif'.format(wctx.out_path, wctx.sizes[1], i.zfill(5)), hr_img)
            tiff.imwrite('{}/sr_{}_{}/{}.tif'.format(wctx.out_path, wctx.sizes[0], wctx.sizes[1], i.zfill(5)), sr_img)
        except Exception as e:
            print(f"Error writing image files for {file.name} (index {i}): {e}. Skipping writes for this file.")
            # Continue to the next file, don't crash the worker
            pass # We pass to continue the loop, as the error is handled

        wctx.inc_get()
    

def all_threads_inactive(worker_threads):
    for thread in worker_threads:
        if thread.is_alive():
            return False
    return True

def prepare(img_path, out_path, n_worker, sizes=(256, 512), resample=Image.BICUBIC, lmdb_save=False):
    resize_fn = partial(resize_worker, sizes=sizes,
                        resample=resample, lmdb_save=lmdb_save)
    files = [p for p in Path(
        '{}'.format(img_path)).glob(f'**/*') if p.is_file()] # Ensure it's a file

    os.makedirs(out_path, exist_ok=True)
    # os.makedirs(r'{}/lr_{}'.format(out_path, sizes[0]), exist_ok=True)
    os.makedirs(r'{}/hr_{}'.format(out_path, sizes[1]), exist_ok=True)
    os.makedirs(r'{}/sr_{}_{}'.format(out_path, sizes[0], sizes[1]), exist_ok=True)

    if n_worker > 1:
        # prepare data subsets
        multi_env = None

        file_subsets = np.array_split(files, n_worker)
        worker_threads = []
        wctx = WorkingContext(resize_fn, lmdb_save, out_path, multi_env, sizes)

        # start worker processes, monitor results
        for i in range(n_worker):
            proc = Process(target=prepare_process_worker, args=(wctx, file_subsets[i]))
            proc.start()
            worker_threads.append(proc)
        
        total_count = str(len(files))
        while not all_threads_inactive(worker_threads):
            print("\r{}/{} images processed".format(wctx.value(), total_count), end=" ")
            time.sleep(0.1)

        # Ensure all processes have finished before exiting
        for proc in worker_threads:
            proc.join()

    else: # This block is for n_worker = 1, essentially running in the main thread
        # The wctx needs to be initialized even for single worker case to use inc_get etc.
        wctx = WorkingContext(resize_fn, lmdb_save, out_path, None, sizes) 
        total = 0 # This 'total' is not used to update wctx.counter in the single worker case. Consider unifying logic.
        for file in tqdm(files):
            i, imgs = resize_fn(file)
            
            if i is None or imgs is None:
                continue

            lr_img, hr_img, sr_img = imgs
            try:
                tiff.imwrite('{}/lr_{}/{}.tif'.format(wctx.out_path, wctx.sizes[0], i.zfill(5)), lr_img)
                tiff.imwrite('{}/hr_{}/{}.tif'.format(wctx.out_path, wctx.sizes[1], i.zfill(5)), hr_img)
                tiff.imwrite('{}/sr_{}_{}/{}.tif'.format(wctx.out_path, wctx.sizes[0], wctx.sizes[1], i.zfill(5)), sr_img)
            except Exception as e:
                print(f"Error writing image files for {file.name} (index {i}): {e}. Skipping writes for this file.")
                pass
            wctx.inc_get() # Increment counter even in single worker mode

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', '-p', type=str,
                        default='{}/Dataset/celebahq_256'.format(Path.home()))
    parser.add_argument('--out', '-o', type=str,
                        default='./dataset/celebahq')

    parser.add_argument('--size', type=str, default='64,512')
    parser.add_argument('--n_worker', type=int, default=3)
    parser.add_argument('--resample', type=str, default='bicubic')
    parser.add_argument('--lmdb', '-l', action='store_true')

    args = parser.parse_args()

    resample_map = {'bilinear': Image.BILINEAR, 'bicubic': Image.BICUBIC}
    resample = resample_map[args.resample]
    sizes = [int(s.strip()) for s in args.size.split(',')]

    args.out = '{}_{}_{}'.format(args.out, sizes[0], sizes[1])
    print(f"Preparing dataset from {args.path} to {args.out} with sizes {sizes} and resample method {args.resample}")
    prepare(args.path, args.out, args.n_worker,
            sizes=sizes, resample=resample, lmdb_save=args.lmdb)