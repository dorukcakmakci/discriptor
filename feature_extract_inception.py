import os
import pdb
import pickle

import pandas as pd
import numpy as np

from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.models import Model
from keras import backend as K

from tqdm import tqdm

inception = InceptionV3(weights='imagenet', include_top=True)
model = Model(inception.input, inception.layers[-2].output)

def extract_features(img):
    #img are 299x299x3 images of rgb
    img = preprocess_input(img)
    img = np.expand_dims(img,axis=0)
    res = model.predict(img)
    res = np.reshape(res, res.shape[1])
    
    return res

# import processed images
with open("./data/dataset/dataset_inception.pkl", "rb") as f:
    dct = pickle.load(f)

images = list(dct.values())
ids = list(dct.keys())

dataset = {}
for (img, id) in tqdm(zip(images, ids)):
    feature = extract_features(img)
    dataset[id] = feature

# save image features
with open("./data/dataset/processed_dataset_inception.pkl", "wb") as f:
    pickle.dump(dataset, f)
