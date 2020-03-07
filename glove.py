import pdb
import h5py
import pickle 

import numpy as np

with h5py.File('./data/eee443_project_dataset_train.h5', 'r') as f:
    captions = f['train_cap'][:]
    image_ids = f['train_imid'][:]
    images = f['train_ims'][:]
    urls = f['train_url'][:]
    word_codes = f['word_code'][:]

glove_embedding_dict = {}
with open("./utils/glove.6B.200d.txt", 'r', encoding="utf-8") as f:
    for line in f:
        values = line.split()
        word = values[0]
        vector = np.asarray(values[1:], "float32")
        glove_embedding_dict[word] = vector

# check if all vocabulary is present on glove embeddings
with open("./utils/word_to_idx.pkl", "rb") as f:
    word_to_idx = pickle.load(f)
    vocabulary = list(word_to_idx.keys())

embedding_dict = {}
error_cnt = 0
successful_cnt = 0
for word in vocabulary:
    try:
        # print(word, "\t", glove_embedding_dict[word].shape)
        embedding_dict[word] = glove_embedding_dict[word]
        successful_cnt += 1
    except:
        print("glove embedding not found for: ", word)
        print("replaced with zeros")
        embedding_dict[word] = np.zeros(200)
        error_cnt += 1


print("Error count: ", error_cnt)
print("Successful count: ", successful_cnt)


embedding_matrix_size = len(list(embedding_dict.keys()))
embedding_matrix = np.zeros((embedding_matrix_size, 200))
for word in embedding_dict.keys():
    embedding_matrix[int(word_to_idx[word]), :] = embedding_dict[word]

# set embedding of unknown tag as average
# embedding_matrix[int(word_to_idx["x_UNK_"]), :] = np.mean(np.array(list(embedding_dict.values())), axis=0)

np.save("./utils/embedding_matrix.npy", embedding_matrix)