import os, random
from shutil import copy

# specify the folders
src_folder = 'images/rawdata4/utkcropped'
dst_folder = 'images/milestone4/test'

all_files_in_folder = os.listdir(src_folder)


random.seed(42)

selected_files = random.sample(all_files_in_folder, 651)

# copy the selected files to the destination folder
for file in selected_files:
    copy(os.path.join(src_folder, file), dst_folder)