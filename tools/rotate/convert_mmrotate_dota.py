import glob
import os
import os.path as osp
from shutil import copy
from tqdm import tqdm

if __name__ == '__main__':
    source_path = '/data/Aerial/DOTA1_0/standard/test/images_'
    dst_path = '/data/Aerial/DOTA1_0/standard/test/images'

    os.makedirs(dst_path, exist_ok=True)
    img_list = glob.glob(source_path+'/*')
    for source_img in img_list:
        base_name = osp.basename(source_img)
        chmod = base_name[:8] + base_name[11:]
        dst_img_name = osp.join(dst_path, chmod)
        copy(source_img, dst_img_name)
        print(source_img + '  -->  '+ dst_img_name)
