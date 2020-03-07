# from tensorflow.python.framework import ops
# ops.reset_default_graph()

import numpy as np
import tensorflow as tf

from keras.layers import Conv1D
from keras.models import Sequential
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.optimizers import Adam, RMSprop
from keras.layers.merge import add
from keras.layers import Input
from keras.layers import Dropout, Dense, LSTM, Embedding
from keras.models import Model, load_model
import keras.backend as K

import pickle
import pdb

def preprocess_dataset(dct):
    result = {}
    for key, value in zip(dct['keys'], dct['values']):
        result[key] = value
    return result

# # all dataset
# with open("./utils/processed_dataset_vgg.pkl", "rb") as f:
#     all_features = pickle.load(f)
# with open("./utils/captions.pkl", "rb") as f:
#     all_captions = pickle.load(f)

# train dataset (70% of the data)
with open("./data/dataset/train_dataset_inception.pkl", "rb") as f:
    train_features = pickle.load(f)
    train_encoded_images = preprocess_dataset(train_features)
with open("./data/dataset/train_captions_inception.pkl", "rb") as f:
    train_captions = pickle.load(f)
    train_dataset = train_captions
    # train_dataset = preprocess_dataset(train_captions)

# validation dataset (20% of the data)
with open("./data/dataset/validation_dataset_inception.pkl", "rb") as f:
    validation_features = pickle.load(f)
    validation_encoded_images = preprocess_dataset(validation_features)
with open("./data/dataset/validation_captions_inception.pkl", "rb") as f:
    validation_captions = pickle.load(f)
    validation_dataset = validation_captions
    # validation_dataset = preprocess_dataset(validation_captions)

# test dataset (10% of the data)
with open("./data/dataset/test_dataset_inception.pkl", "rb") as f:
    test_features = pickle.load(f)
    test_encoded_images = preprocess_dataset(test_features)
with open("./data/dataset/test_captions_inception.pkl", "rb") as f:
    test_captions = pickle.load(f)
    test_dataset = test_captions
    # test_dataset = preprocess_dataset(test_captions)

# train_dataset = train_captions
# train_encoded_images = train_features 

# test_dataset = test_captions
# test_encoded_images = test_features

# validation_dataset = validation_captions
# validation_encoded_images = validation_features


# model
max_length = 17
embedding_dim = 200
vocabulary_size = 1004

'''
Model takes 2 types of inputs, first being encoded photos by inception to 2048 dimensional vectors. Second input type to the model are sequences of words (i.e, captions).
Captions are all zero padded to max_length. With first item being SOS token and last item being EOS token. Dropout layers are used for prevention of overfitting. Photos are processed
such that their dimensions are reduced to 256 by dense layer with relu activation. Caption sequences are processed by the glove pretrained embedding matrix. Embedding dimension
is chosen as 200. Vocabulary size in the problem is 1020.
'''

inputs1 = Input(shape=(2048,))
features1 = Dropout(0.5)(inputs1)
features2 = Dense(256, activation='relu')(features1)
inputs2 = Input(shape=(max_length,))
seqfeatures1 = Embedding(vocabulary_size, embedding_dim, mask_zero=True)(inputs2)
seqfeatures2 = Dropout(0.5)(seqfeatures1)
seqfeatures3 = LSTM(256)(seqfeatures2)

#add two different types of feature extractions into one tensor.
decoder1 = add([features2, seqfeatures3])

decoder2 = Dense(256, activation='relu')(decoder1)
#softmax to predict target word over vocabulary. 
outputs = Dense(vocabulary_size, activation='softmax')(decoder2)
#construction of the model.
model = Model(inputs=[inputs1, inputs2], outputs=outputs)


'''
Data generator function streams the input data to the model in batches. This will be used for fit_generator function with the model.
Note that, batch size is in unit of images. With batch_size = 3, there will be 3 images with corresponding training sample generation. 
'''

batch_size = 6
def data_generator(dataset, imgs, max_length, batch_size = batch_size):
    X1, X2, y = list(), list(), list()
    n = 0
    #infinite loop over images
    while True:
        for key, captions_list in dataset.items():
            n += 1
            # retrieve the photo feature
            try:
                image = imgs[key]
            except:
                n -= 1
                continue
            for capt in captions_list:
                # split one sequence into multiple X, y pairs
                for i in range(1, len(capt)):
                    # split into input and output pair
                    in_seq, out_seq = capt[:i], capt[i]
                    if out_seq == 0:
                        break
                    # pad input sequence
                    in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
                    # encode output sequence
                    out_seq = to_categorical([out_seq], num_classes=vocabulary_size)[0]
                    # store
                    X1.append(image)
                    X2.append(in_seq)
                    y.append(out_seq)
            # yield the batch data
            if n==batch_size:
                yield [[np.array(X1), np.array(X2)], np.array(y)]
                X1, X2, y = list(), list(), list()
                n=0

#get embedding matrix with shape 1024x200
print(model.summary())
embedding_matrix = np.load("./utils/embedding_matrix.npy")
model.layers[2].set_weights([embedding_matrix])
model.layers[2].trainable = False

#compile the model with categorical cross entropy, combined with softmax
model.compile(loss='categorical_crossentropy', optimizer='adam')

steps = len(train_dataset)//batch_size
# generator = data_generator(train_dataset, train_encoded_images, max_length, batch_size)
# model.optimizer.lr = 0.01
# model.fit_generator(generator, epochs=9, steps_per_epoch=steps, verbose=1)
# model.save('./model_' + '9epochs' + '.h5')
epochs = 50
# # load model
base = 0
# model = load_model('./model_weights/model_9.h5')
# model.optimizer.learning_rate = 0.00001

# print(K.eval(model.optimizer.lr))
# K.set_value(model.optimizer.lr, 0.000001)
# print(K.eval(model.optimizer.lr))


for i in range(1,epochs+1):
    if i == 5:
        batch_size = 12
        K.set_value(model.optimizer.lr, 0.0001)
    elif i == 10:
        batch_size = 18
        K.set_value(model.optimizer.lr, 0.00001)
    train_generator = data_generator(train_dataset, train_encoded_images, max_length, batch_size)
    validation_generator = data_generator(validation_dataset, validation_encoded_images, max_length, batch_size)
    # pdb.set_trace()
    model.fit_generator(train_generator, epochs=1, steps_per_epoch=steps, verbose=1, validation_data=validation_generator, validation_steps=steps, validation_freq=1)
    model.save('./model_weights/model_inception_epoch' + str(base+i) + '.h5')


