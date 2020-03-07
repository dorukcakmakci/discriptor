
import pickle
from keras.preprocessing.sequence import pad_sequences
import numpy as np 
import pdb
import random
from keras.applications.inception_v3 import InceptionV3, preprocess_input
import sys
from PIL import Image
from keras.models import load_model
from tensorflow.keras.models import Model

# import conversion dictionaries
with open("./utils/word_to_idx.pkl", "rb") as f:
    word_to_idx = pickle.load(f)
with open("./utils/idx_to_word.pkl", "rb") as f:
    idx_to_word = pickle.load(f)

    # all dataset
with open("./data/dataset/processed_dataset_vgg.pkl", "rb") as f:
    features = pickle.load(f)
with open("./utils/captions.pkl", "rb") as f:
    captions = pickle.load(f)

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
    result = [idx_to_word[str(x)] for x in caption]
    result = result[1:-1]
    result = ' '.join(result)
    return result

images = [24182, 137, 30296, 10518, 25178, 16968, 21062, 30707, 27506, 18408, 39625, 860,6086, 40825, 8690, 32552]

for id in images:
    image = Image.open(os.path.join('./data/dataset/images/'+str(id)+".jpeg"))
    img, cap = get_image(id)
    pred_cap = describe_image(features[str(id)], model)
    pdb.set_trace()

    