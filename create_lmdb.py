import argparse
import os
import os.path as osp
import pickle
import random

import lmdb
import torch.utils.data as data
import torchvision.transforms as tfs
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.transforms import functional as FF
from data_utils import LMDB_Image

from metrics import *

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', help='name of dataset', choices=['indoor', 'outdoor', 'RICE'],
                    default='RICE')
parser.add_argument('--n1_path', type=str, default='/content/drive/MyDrive/NCKHSV/C2PNet/6Models/AIDTransformer/results/Dehazed/Sate1K_Thick', help='negative path')
# parser.add_argument('--n2_path', type=str, default='/content/drive/MyDrive/NCKHSV/C2PNet/6Models/DCP/DCPNet', help='negative path')
# parser.add_argument('--n3_path', type=str, default='/content/drive/MyDrive/NCKHSV/C2PNet/6Models/Dehamer/DeHarmer', help='negative path')
# parser.add_argument('--n4_path', type=str, default='/content/drive/MyDrive/NCKHSV/C2PNet/6Models/FFA-Net/FFA', help='negative path')
# parser.add_argument('--n5_path', type=str, default='/content/drive/MyDrive/NCKHSV/C2PNet/6Models/GCANet/GCA', help='negative path') #FLAG
# parser.add_argument('--n6_path', type=str, default='/content/drive/MyDrive/NCKHSV/C2PNet/6Models/GridDehazeNet/GDN', help='negative path')
parser.add_argument('--dpath', type=str, default='/content/drive/MyDrive/NCKHSV/C2PNet/C2PNet/data/SATE1K_Thick', help='LMDB save path')
parser.add_argument('--name', type=str, default='SATE1K_Thick', help='LMDB name')

# parser = argparse.ArgumentParser()
# parser.add_argument('--dataset_name', help='name of dataset', choices=['indoor', 'outdoor', 'RICE'],
#                     default='RICE')
# parser.add_argument('--n1_path', type=str, default='/content/drive/MyDrive/NCKHSV/C2PNet/C2PNet/data/RICE_train/test/hazy', help='negative path')
# parser.add_argument('--n2_path', type=str, default='/content/drive/MyDrive/NCKHSV/C2PNet/C2PNet/data/RICE_train/test/hazy', help='negative path')
# parser.add_argument('--n3_path', type=str, default='/content/drive/MyDrive/NCKHSV/C2PNet/C2PNet/data/RICE_train/test/hazy', help='negative path')
# parser.add_argument('--n4_path', type=str, default='/content/drive/MyDrive/NCKHSV/C2PNet/C2PNet/data/RICE_train/test/hazy', help='negative path')
# parser.add_argument('--n5_path', type=str, default='/content/drive/MyDrive/NCKHSV/C2PNet/C2PNet/data/RICE_train/test/hazy', help='negative path') #FLAG
# parser.add_argument('--n6_path', type=str, default='/content/drive/MyDrive/NCKHSV/C2PNet/C2PNet/data/RICE_train/test/hazy', help='negative path')
# parser.add_argument('--dpath', type=str, default='/content/drive/MyDrive/NCKHSV/C2PNet/C2PNet/data/RICE', help='LMDB save path')
# parser.add_argument('--name', type=str, default='RICE', help='LMDB name')

opt = parser.parse_args()


class RESIDE_Dataset_C2R(data.Dataset):
    def __init__(self, path, size='whole img', format='.png'):
        super(RESIDE_Dataset_C2R, self).__init__()
        self.size = size
        print('crop size (1)', size)
        self.format = format
        self.haze_imgs_dir = os.listdir(os.path.join(path, 'hazy'))
        self.haze_imgs = [os.path.join(path, 'hazy', img) for img in self.haze_imgs_dir]
        self.clear_dir = os.path.join(path, 'clear')

    def __getitem__(self, index):
        haze = Image.open(self.haze_imgs[index])
        if isinstance(self.size, int):
            while haze.size[0] < self.size or haze.size[1] < self.size:
                index = random.randint(0, 20000)
                haze = Image.open(self.haze_imgs[index])
        img = self.haze_imgs[index]
        haze_name = img.split('/')[-1]
        # print(haze_name)
        id = haze_name.split('_')[0]
        clear_name = id #FLAG
        clear = Image.open(os.path.join(self.clear_dir, clear_name))
        n1 = Image.open(os.path.join(opt.n1_path, haze_name))
        # n2 = Image.open(os.path.join(opt.n2_path, haze_name))
        # n3 = Image.open(os.path.join(opt.n3_path, haze_name))
        # n4 = Image.open(os.path.join(opt.n4_path, haze_name))
        # n5 = Image.open(os.path.join(opt.n5_path, haze_name))
        # n6 = Image.open(os.path.join(opt.n6_path, haze_name))
        clear = tfs.CenterCrop(haze.size[::-1])(clear)
        # out = self.aug_data(haze, clear, n1, n2, n3, n4, n5, n6)
        out = self.aug_data(haze, clear, n1)
        return out

    def aug_data(self, *images):
        out = []
        for img in images:
            out.append(np.array(img, dtype=np.uint8))
        return out

    def __len__(self):
        return len(self.haze_imgs)


def data2lmdb(dpath, name="train", write_frequency=5, num_workers=2): #FLAG
    dataset = RESIDE_Dataset_C2R('/content/drive/MyDrive/NCKHSV/Dehaze_Dataset/SATE1K/Haze1k/Haze1k_thick/dataset/train', size='whole img')
    data_loader = DataLoader(dataset, num_workers=2, collate_fn=lambda x: x)
    print("length of data", len(data_loader)) #FLAG
    lmdb_path = osp.join(dpath, "%s.lmdb" % name)
    isdir = os.path.isdir(lmdb_path)

    print("Generate LMDB to %s" % lmdb_path)
    db = lmdb.open(lmdb_path, subdir=isdir,
                   map_size=1099511627776 * 2, readonly=False,
                   meminit=False, map_async=True)
    txn = db.begin(write=True)

    for idx, data in enumerate(data_loader):

        # haze, clear, n1, n2, n3, n4, n5, n6 = data[0]
        # temp = LMDB_Image(haze, clear, n1, n2, n3, n4, n5, n6)
        haze, clear, n1 = data[0]
        temp = LMDB_Image(haze, clear, n1)        
        txn.put(u'{}'.format(idx).encode('ascii'), pickle.dumps(temp))
        # print("FLAG")
        if idx % write_frequency == 0:
            print(idx)
            print("[%d/%d]" % (idx, len(data_loader)))
            txn.commit()
            txn = db.begin(write=True)

    # finish iterating through dataset
    txn.commit()
    keys = [u'{}'.format(k).encode('ascii') for k in range(idx + 1)]
    with db.begin(write=True) as txn:
        txn.put(b'__keys__', pickle.dumps(keys))
        txn.put(b'__len__', pickle.dumps(len(keys)))

    print("Flushing database ...")
    db.sync()
    db.close()


if __name__ == "__main__":
    data2lmdb(opt.dpath, opt.name)
