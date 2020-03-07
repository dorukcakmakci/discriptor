import pickle
import os 
from nltk.translate.bleu_score import corpus_bleu

# test dataset (10% of the data)
with open("./data/dataset/test_dataset_inception.pkl", "rb") as f:
    test_features = pickle.load(f)
    test_encoded_images = preprocess_dataset(test_features)
with open("./data/dataset/test_captions_inception.pkl", "rb") as f:
    test_captions = pickle.load(f)
    test_dataset = test_captions

# relevant dictionaries for word conversion
with open("./utils/word_to_idx.pkl", "rb") as f:
    word_to_idx = pickle.load(f)
with open("./utils/idx_to_word.pkl", "rb") as f:
    idx_to_word = pickle.load(f)

# other 
inception = InceptionV3(weights='imagenet', include_top=True)
inception = Model(inception.input, inception.layers[-2].output)
model = load_model('./model_weights/model_inception_epoch1.h5')
max_length = 17
corpus_bleu_score = 0
references = []
candidates= []
for key in test_captions.keys():
    candidate = ['x_START_']
    reference = test_captions[key]
    for i in range(max_length):
        sequence = [word_to_idx[w] for w in candidate if w in word_to_idx.keys()]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = model.predict([img,sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = idx_to_word[str(yhat)]
        candidate.append(word)
        if word == 'x_END_':
            while len(candidate) != 17:
                candidate.append('x_NULL_')
            break
    candidates.append(candidate)
    references.append(reference)

bleu = corpus_bleu(references, candidates)
print("Bleu Score: ", bleu)
    
    