import h5py
import os
import pdb
import urllib.request

import numpy as np

from multiprocessing.pool import ThreadPool



with h5py.File('./data/eee443_project_dataset_train.h5', 'r') as f:
    captions = f['train_cap'][:]
    image_ids = f['train_imid'][:]
    images = f['train_ims'][:]
    urls = f['train_url'][:]
    word_codes = f['word_code'][:]

print("Dataset Fields")
print("Captions: ", captions.shape)
print("Image IDs: ", image_ids.shape)
print("Images: ", images.shape)
print("URLs: ", urls.shape)
print("Word Codes: ", word_codes.shape)
print("Min image id:", np.min(image_ids))
print("Max image id:", np.max(image_ids))

dataset_path = "./data/dataset/"
images_path = os.path.join(dataset_path, "images/")

# first create a directory named "dataset/images/" to be used as download destination 
# uncomment to download images
# error_count = 0
urls = list(enumerate(urls))
def download(idx, url):
    try:
        urllib.request.urlretrieve(url.decode('UTF-8'), os.path.join('./data/dataset/images/', str(idx + 1) + ".jpg"))
    except urllib.request.HTTPError:
        pass
with ThreadPool(20) as p:
    p.starmap(download, urls)


# for idx, url in enumerate(urls):
#     try:
#         urllib.request.urlretrieve(url.decode('UTF-8'), os.path.join(images_path, str(idx) + ".jpg"))
#     except urllib.request.HTTPError as e:
#         error_count += 1
#         print("Error occured at idx: ", idx, " with source: ", url)
#         print("Error Status Code: ", e.code)
#         print("Error Message: ", e.read())

# print(error_count, "images were not downloaded due to errors.")