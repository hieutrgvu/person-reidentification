import argparse
import os
from shutil import copyfile

parser = argparse.ArgumentParser(description='Prepare data')
parser.add_argument('--data-path', default='./market', type=str, help='data path')
opt = parser.parse_args()

save_path = opt.data_path + '/pytorch'
if not os.path.isdir(save_path):
    os.mkdir(save_path)

redirect = {
    'query': 'query',
    'gt_bbox': 'multi-query',
    'bounding_box_test': 'gallery',
    'bounding_box_train': 'train_all',
}

for (src, dest) in redirect.items():
    print('preparing', dest, "...")
    src_path = opt.data_path + '/' + src
    dst_path = save_path + '/' + dest

    if not os.path.isdir(dst_path):
        os.mkdir(dst_path)

    for (root, dirs, files) in os.walk(src_path, topdown=True):
        for name in files:
            if not name[-3:] == 'jpg':
                continue
            ID = name.split('_')
            person_src_path = src_path + '/' + name
            person_dst_path = dst_path + '/' + ID[0]
            if not os.path.isdir(person_dst_path):
                os.mkdir(person_dst_path)
            copyfile(person_src_path, person_dst_path + '/' + name)

# train and val set
print('prepare train and validation set ...')
train_path = opt.data_path + '/bounding_box_train'
train_save_path = save_path + '/train'
val_save_path = save_path + '/val'
if not os.path.isdir(train_save_path):
    os.mkdir(train_save_path)
    os.mkdir(val_save_path)

for root, dirs, files in os.walk(train_path, topdown=True):
    for name in files:
        if not name[-3:] == 'jpg':
            continue
        ID = name.split('_')
        src_path = train_path + '/' + name
        dst_path = train_save_path + '/' + ID[0]
        if not os.path.isdir(dst_path):
            os.mkdir(dst_path)
            # first image is used as val image
            dst_path = val_save_path + '/' + ID[0]
            os.mkdir(dst_path)
        copyfile(src_path, dst_path + '/' + name)

print('all done!')
