import pickle
import h5py
import os
import pdb

import numpy as np

from gensim.models import Word2Vec

# input is a list of integers or words, and corresponding lookup table
def convert_caption(caption, dct):
    result = []
    for item in caption:
        result.append(dct[str(item)])
    return result

# import dataset
with h5py.File('./data/eee443_project_dataset_train.h5', 'r') as f:
    captions = f['train_cap'][:]
    word_codes = f['word_code'][:]
    image_ids = f['train_imid'][:]

vocabulary_size = len(word_codes[0].dtype)

# create dictionaries for conversion between words and corresponding indices 
idx_to_word = {}
word_to_idx = {}
for i in range(vocabulary_size):
    idx_to_word[str(int(word_codes[0][i]))] = word_codes[0].dtype.descr[i][0]
    word_to_idx[word_codes[0].dtype.descr[i][0]] = str(int(word_codes[0][i]))

pdb.set_trace()

# save all dictionaries
with open('./utils/word_to_idx.pkl', 'wb') as f:
    pickle.dump(word_to_idx, f)
with open('./utils/idx_to_word.pkl', 'wb') as f:
    pickle.dump(idx_to_word, f)
