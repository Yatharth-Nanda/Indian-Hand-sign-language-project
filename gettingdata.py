import os
import random
import shutil

# Set the source directory containing the folders
source_directory = r'C:\Users\yatha\Downloads\archive_islset\Indian'

# Set the destination directory where you want to copy the random images
destination_directory = r'C:\Users\yatha\Downloads\archive_islset\Indian_modified'

# Set the number of images you want to select from each folder
num_images_per_folder = 40

# Iterate over each folder
for folder_name in os.listdir(source_directory):
    folder_path = os.path.join(source_directory, folder_name)

    # Check if it's a directory
    if os.path.isdir(folder_path):
        # Get list of images in the folder
        images = os.listdir(folder_path)

        # Randomly select num_images_per_folder images
        selected_images = random.sample(images, min(num_images_per_folder, len(images)))

        # Create destination folder if it doesn't exist
        destination_folder = os.path.join(destination_directory, folder_name)
        os.makedirs(destination_folder, exist_ok=True)

        # Copy selected images to the destination folder
        for image in selected_images:
            source_image_path = os.path.join(folder_path, image)
            destination_image_path = os.path.join(destination_folder, image)
            shutil.copyfile(source_image_path, destination_image_path)
