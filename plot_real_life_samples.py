
import pickle
from keras.preprocessing.sequence import pad_sequences
import numpy as np 
import pdb
import random
from keras.applications.inception_v3 import InceptionV3, preprocess_input
import sys
from PIL import Image
from keras.models import load_model
from keras.models import Model
import os
import matplotlib.pyplot as plt

# import conversion dictionaries
with open("./utils/word_to_idx.pkl", "rb") as f:
    word_to_idx = pickle.load(f)
with open("./utils/idx_to_word.pkl", "rb") as f:
    idx_to_word = pickle.load(f)

    # all dataset
with open("./data/dataset/processed_dataset_inception.pkl", "rb") as f:
    features = pickle.load(f)
with open("./utils/captions.pkl", "rb") as f:
    captions = pickle.load(f)

count = 0
for key in captions.keys():
    for capt in captions[key]:
        count+=1
pdb.set_trace()
inception = InceptionV3(weights='imagenet', include_top=True)
inception = Model(inception.input, inception.layers[-2].output)

model = load_model('./model_weights/model_inception_epoch1.h5')
max_length = 17

def get_image(image_id):
    feature = features[str(image_id)]
    caps = captions[str(image_id)]
    cap = caps[0] # use first caption
    return feature, cap

# generate caption for a given sample
def describe_image(img, model):
    input_string = 'x_START_'
    for i in range(max_length):
        sequence = [word_to_idx[w] for w in input_string.split() if w in word_to_idx.keys()]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = model.predict([img,sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = idx_to_word[str(yhat)]
        input_string += ' ' + word

        if word == 'x_END_':
            break

    final = input_string.split()
    final = final[1:-1]
    final = ' '.join(final)
    return final

def get_image_caption(image_id):
    features, caption = get_image(image_id)
    result = [idx_to_word[str(x)] for x in caption if idx_to_word[str(x)] != "x_NULL_"]
    result = result[1:-1]
    result = ' '.join(result)
    return result

images = ["car_1", "doruk_1", "ev_1", "furkan_laptop"]

for id in images:
    path= os.path.join('./data/real_life_images/'+id+".jpeg")
    img_ = Image.open(path)
    processed_img = img_.resize((299,299), resample=Image.ANTIALIAS)
    temp = np.array(processed_img)
    if temp.shape == (299,299):
        temp_ = [temp, temp, temp]
        temp = np.stack(temp_, axis=2)
    img= np.array(temp, dtype=np.uint8)
    img = preprocess_input(img)
    img = np.expand_dims(img,axis=0)
    res = inception.predict(img)
    res = np.reshape(res, res.shape[1])
    img_features = np.expand_dims(res, axis=0)

    pred_cap = describe_image(img_features, model)
    print(pred_cap)
    fig, ax = plt.subplots()
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    ax.imshow(img_)
    ax.axis('on')
    ax.set_xlabel("Predicted Caption: " + pred_cap, fontsize=10)
    plt.savefig(id+"_capt")
    plt.close()
    

    