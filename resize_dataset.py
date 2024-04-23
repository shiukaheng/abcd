# Disable user warnings
import warnings
warnings.filterwarnings('ignore')

# Index datasets
import os
import shutil
from PIL import Image
import logging
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

import argparse

def process_image(args):
    """
    Function to be executed by each process.
    """
    source_path, destination_path, max_mp, filename = args

    try:
        # Open the image
        with Image.open(source_path) as img:
            width, height = img.size
            # Calculate the current megapixels of the image
            current_mp = (width * height) / 1_000_000

            # If the image size exceeds the maximum megapixels, resize it
            if current_mp > max_mp:
                scale_factor = (max_mp / current_mp) ** 0.5  # Square root of ratio to scale both dimensions
                new_size = (int(width * scale_factor), int(height * scale_factor))
                resized_img = img.resize(new_size, Image.Resampling.LANCZOS)
                # Save the resized image to the destination folder
                resized_img.save(destination_path)
                return f"Resized and saved {filename} to {destination_path}"
            else:
                # If the image does not exceed the max size, copy it to the destination folder
                shutil.copy2(source_path, destination_path)
                return f"Copied {filename} to {destination_path}"
    except Exception as e:
        logging.error(f"Failed to process {filename}: {e}")
        return f"Failed to process {filename}: {e}"

def direct_resize_images_with_progress(source_folder, destination_folder, max_mp):
    # Create the destination folder if it doesn't exist
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    # Prepare arguments for multiprocessing
    tasks = []
    for filename in os.listdir(source_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
            source_path = os.path.join(source_folder, filename)
            destination_path = os.path.join(destination_folder, filename)
            tasks.append((source_path, destination_path, max_mp, filename))
    
    # Number of processes
    num_processes = min(cpu_count(), len(tasks))  # Don't spawn more processes than tasks

    # Execute the tasks in parallel
    with Pool(num_processes) as pool:
        for result in tqdm(pool.imap_unordered(process_image, tasks), total=len(tasks), desc="Processing images"):
            logging.info(result)

if __name__ == "__main__":
    # Minimalist cli for converting image dataset sizes.
    # Provide dataset path: python resize_dataset.py ./dataset --max-mp 2
    # Original dataset structure:
    # ./images
        # <image1>.jpg
        # ...
    # ./sparse

    # We want to rename images to images_original, and resize the images to the images folder.
    # If images_original already exists, we reprocess from image_original to images with new max_mp into images.

    parser = argparse.ArgumentParser(description="Resize images in a dataset to a maximum number of megapixels.")
    parser.add_argument("dataset_path", type=str, help="Path to the dataset folder.")
    parser.add_argument("--max-mp", type=float, default=2, help="Maximum number of megapixels for each image.")
    args = parser.parse_args()

    dataset_path = args.dataset_path
    max_mp = args.max_mp

    # Check if the dataset path exists
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset path {dataset_path} not found.")
    
    # Check if the dataset path contains an images folder
    images_folder = os.path.join(dataset_path, "images")

    if not os.path.exists(images_folder):
        raise FileNotFoundError(f"Images folder not found in dataset path {dataset_path}.")
    
    # Check if the dataset path contains a sparse folder
    sparse_folder = os.path.join(dataset_path, "sparse")

    if not os.path.exists(sparse_folder):
        raise FileNotFoundError(f"Sparse folder not found in dataset path {dataset_path}.")
    
    # Check if the images_original folder exists
    images_original_folder = os.path.join(dataset_path, "images_original")
    if not os.path.exists(images_original_folder):
        # Rename the images folder to images_original
        os.rename(images_folder, images_original_folder)

    # Resize the images in the images_original folder
    direct_resize_images_with_progress(images_original_folder, images_folder, max_mp)

    logging.info("Image resizing completed.")



