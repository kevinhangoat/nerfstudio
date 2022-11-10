import os   
import pdb
import glob

path = "/srv/beegfs02/scratch/bdd100k/data/sfm/nerfstudio/renders/d389c316-c71f7a5e/test_depth_transient/"
png_counter = 0
jpeg_counter = 0
list_of_files = sorted(glob.glob(path + '*'))
for img in list_of_files:
    if img[-3:] == 'png':
        png_counter += 1
        old_img_path = img
        new_name = f'{png_counter:05d}.png'
        new_img_path = os.path.join(path, new_name)
    elif img[-4:] == 'jpeg':
        jpeg_counter += 1
        old_img_path = img
        new_name = f'{jpeg_counter:05d}.jpeg'
        new_img_path = os.path.join(path, new_name)
    os.system(f"mv {old_img_path} {new_img_path}")
