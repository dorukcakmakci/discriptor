import os
import pdb
import h5py
import pickle

import numpy as np 

from tqdm import tqdm

with h5py.File('./data/eee443_project_dataset_train.h5', 'r') as f:
    captions = f['train_cap'][:]
    image_ids = f['train_imid'][:]

min_id = np.min(image_ids)
max_id = np.max(image_ids)

dct = {}
for i in tqdm(range(min_id, max_id + 1)):
    idx = np.where(image_ids == i)
    dct[str(i)] = [x.tolist() for x in captions[idx]]

pdb.set_trace()

with open("./utils/captions.pkl", "wb") as f:
    pickle.dump(dct, f)
    