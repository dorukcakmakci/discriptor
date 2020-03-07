import os
import pdb
import pickle

import pandas as pd
import numpy as np

from keras.applications.vgg16 import VGG16, preprocess_input
from keras.models import Model
from tqdm import tqdm
from keras import backend as K


vgg = VGG16(weights='imagenet', include_top=True)
transfer_layer = vgg.get_layer('fc2') # one previous from last
model = Model(vgg.input, transfer_layer.output)

def extract_features(img):
    #img are 299x299x3 images of rgb
    img = preprocess_input(img)
    img = np.expand_dims(img,axis=0)
    res = model.predict(img)
    res = np.reshape(res, res.shape[1])
    return res

# import processed images
with open("./data/dataset/dataset_vgg.pkl", "rb") as f:
    dct = pickle.load(f)

images = list(dct.values())
ids = list(dct.keys())

dataset = {}
for (img, id) in tqdm(zip(images, ids)):
    feature = extract_features(img)
    dataset[id] = feature

# save image features
with open("./data/dataset/processed_dataset_vgg.pkl", "wb") as f:
    pickle.dump(dataset, f)
