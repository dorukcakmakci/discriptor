# Discriptor

Image Caption Generator Based on Deep Learning

## Description of the Scripts

##### download_images.py

Downloads all of the images in the dataset using 20 threads to ./data/dataset/images/ directory. Uses .h5 file given for the project.

##### process_images_inception.py:

Preprocesses all the downloaded images in ./data/dataset/images/ directory to make them 299x299 RGB images(For Inception_v3 feature extraction). Produces ./data/dataset/dataset_inception.pkl

##### feature_extract_inception.py: 

Extracts feature using InceptionV3 from preprocessed images located in ./data/dataset/dataset_inception.pkl. Writes output to ./data/dataset/processed_dataset_inception.pkl

##### process_images_vgg.py:

Prerocesses all the downloaded images in ./data/dataset/images/ directory to make them 224x224 RGB images(For VGG16 feature extraction). Produces ./data/dataset/dataset_vgg.pkl

##### feature_extract_vgg.py: 

Extracts feature using VGG16 from preprocessed images located in ./data/dataset/dataset_vgg.pkl. Writes output to ./data/dataset/processed_dataset_vgg.pkl

##### produce_captions.py:

Convert captions given in the .h5 file to a list of lists for each image. Writes output to ./utils/captions.pkl

##### create_dictionary.py:

Creates word_to_idx and idx_to_word dictionaries which are used to convert between word indexes(integer) and words themselves. The dictionaries are saved to ./utils/word_to_idx.pkl and ./utils/idx_to_word.pkl

##### glove<span></span>.py: 

Creates embedding matrix from glove pretrained word vectors. Stores embedding matrix to ./utils/embedding_matrix.npy

##### split_dataset_inception.py:

splits the extracted features to train, validation and test datasets for inception model. Outputs written to: 
* ./data/dataset/train_dataset_inception.pkl
* ./data/dataset/validation_dataset_inception.pkl
* ./data/dataset/test_dataset_inception.pkl
* ./data/dataset/train_caption_inception.pkl
* ./data/dataset/validation_caption_inception.pkl
* ./data/dataset/test_caption_inception.pkl

##### split_dataset_vgg.py:

splits the extracted features to train, validation and test datasets for inception model. Outputs written to: 
* ./data/dataset/train_dataset_vgg.pkl
* ./data/dataset/validation_dataset_vgg.pkl
* ./data/dataset/test_dataset_vgg.pkl
* ./data/dataset/train_caption_vgg.pkl
* ./data/dataset/validation_caption_vgg.pkl
* ./data/dataset/test_caption_vgg.pkl

##### discriptor_inception.py:

Train our model on Inception-V3 features

##### discriptor_vgg.py:

Train our model on VGG-16 features

##### plot_real_life_samples.py:

Test our model on images we have taken by our cellphone camera

##### plot_samples.py:

Test our model on test set we have created for the project using the given dataset

##### tsne<span></span>.py:

Apply t-SNE on glove word embedding vectors and plot with respect to 2 components

##### word_frequency.py:

Find frequency of words in the dataset

##### bleu<span></span>.py:

Calculate bleu score

## Results and Implementation Details

Refer to report.pdf file for implementation details and results of our model.


## Team Members

Doruk Çakmakçı
Furkan Özden 
Mert Albaba

@Bilkent University
