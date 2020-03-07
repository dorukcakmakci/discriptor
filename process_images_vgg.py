import os
import pdb
import h5py
import re
import pickle

import numpy as np

from PIL import Image
from tqdm import tqdm

with h5py.File('./data/eee443_project_dataset_train.h5', 'r') as f:
    captions = f['train_cap'][:]
    image_ids = f['train_imid'][:]
    images = f['train_ims'][:]
    urls = f['train_url'][:]
    word_codes = f['word_code'][:]

# preprocess and save images
url_count = 82783
filenames = os.listdir('./data/dataset/images/')
image_ids = list(range(1, url_count + 1))
dataset = {}
for filename in tqdm(filenames):
    img = Image.open(os.path.join('./data/dataset/images/', filename))
    img_id = int(re.split('\.', filename)[0])
    try:
        img = img.convert('RGB')
        processed_img = img.resize((224,224), resample=Image.ANTIALIAS) 
        # print(filename, " --> ", img_id)
        # image is greyscale
        temp = np.array(processed_img)
        if temp.shape == (224,224):
            temp_ = [temp, temp, temp]
            temp = np.stack(temp_, axis=2)
        dataset[str(img_id)] = np.array(temp, dtype=np.uint8)
    except:
        print("Exception --> ", img_id)
pdb.set_trace()

# generate training and test datasets
with open('./data/dataset/dataset_vgg.pkl', 'wb') as f:
    pickle.dump(dataset, f)