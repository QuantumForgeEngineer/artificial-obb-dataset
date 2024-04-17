import os

from utils import remove_all_files, generate_images

# Parameters
num_images = 100  # Total number of images to generate
image_size = 512  # Size of each image

writing_path = "/path/where/to/write/data"

# Empty folders
if os.path.exists(writing_path):
  for fol in [os.path.join(writing_path, "images"), os.path.join(writing_path, "labels")]:
      for ds in ['train', 'val', 'test']:
          remove_all_files(os.path.join(fol, ds))
else:
  os.makedirs(os.path.join(writing_path, "images"))
  os.makedirs(os.path.join(writing_path, "labels"))
  for fol in [os.path.join(writing_path, "images"), os.path.join(writing_path, "labels")]:
      for ds in ['train', 'val', 'test']:
          os.makedirs(os.path.join(fol, ds))
        
# Generate images and labels
generate_images(writing_path, num_images, image_size)
