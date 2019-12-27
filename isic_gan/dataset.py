import urllib.request
import json
import os
from pathlib import Path

from PIL import Image
import numpy as np

from random import random, choice, shuffle

from joblib import Parallel, delayed

import logging

DATASET_BASE_FOLDER = 'dataset'
DATASET_LOCAL_PATH = 'dataset/raw'
DATASET_LOCAL_INDEX_PATH = 'dataset/raw/index.json'
DATESET_TRAIN_TEST_INDEX_PATH = 'train_test_index.json'
DATASET_NOT_USED_INDEX_PATH = 'not_used_imgs_index.json'
ISIC_API_URL = 'https://isic-archive.com/api/v1/'

CLASSES = ['nevus',
            'melanoma',
            'seborrheic keratosis',
            'lichenoid keratosis',
            'dermatofibroma',
            'angioma',
            'basal cell carcinoma',
            None]

if not Path(DATASET_BASE_FOLDER).exists():
    os.mkdir(DATASET_BASE_FOLDER)
if not Path(DATASET_LOCAL_PATH).exists():
    os.mkdir(DATASET_LOCAL_PATH)

def save_index(content, index_path=DATASET_LOCAL_INDEX_PATH):
    content = json.dumps(content)
    try:
        file = open(index_path, 'wt')
        file.write(content)
        file.close()
    except:
        print('[ERROR]: Oops, something really wrong happend saving the index file.')

def load_index(index_path=DATASET_LOCAL_INDEX_PATH):
    try:
        file = open(index_path, 'rt')
        content = file.read()
        file.close()
    except OSError:
        content = {}
        save_index(content, index_path)
        content = '{}'
    finally:
        return json.loads(content)

def get_image_list(imgs_to_list=100, offset=0):
    try:
        url = urllib.request.urlopen(ISIC_API_URL+'image?limit=%d&offset=%d' % (imgs_to_list, offset))
        return json.loads(url.read().decode())
    except:
        print('[ERROR]: Oops, something really bad happend at downloading the image list with offset %d' % offset)
        return []

def get_image_details(id):
    try:
        url = urllib.request.urlopen(ISIC_API_URL+'image/%s' % id)
        return json.loads(url.read().decode())
    except:
        print('[ERROR]: Oops, something really bad happend at downloading the image details with id %s' % id)
        return None

def fetch_image(id):
    try:
        url =  urllib.request.urlopen(ISIC_API_URL+'image/%s/download?contentDisposition=inline' % id)
        return url.read()
    except:
        print('[ERROR]: Oops, something really bad happend at downloading the image with id %s' % id)
        return b''

def save_img(fileName, data):
    try:
        file = open('%s/%s.jpg' % (DATASET_LOCAL_PATH, fileName), "wb")
        file.write(data)
        file.close()
        return True
    except:
        print('[ERROR]: Oops, something really bad happend at saving a image')
        return False

def download_imgs(imgs=100, start_offset=0):
    index = load_index()
    index_keys = list(index.keys())
    downloaded = 0
    current_offset = start_offset
    imgs_to_list = 200

    print('Downloading imgs...')

    while True:

        if downloaded >= imgs:
            break

        for img_ref in get_image_list(imgs_to_list, current_offset):

            id = img_ref['_id']

            if id in index_keys:
                continue # Ignore images that are already at index

            img_details = get_image_details(id)
            if not img_details:
                continue # Ignore if can't get image details

            # Get the important data
            try:
                image_details_temp = {}
                image_details_temp['name'] = img_details['name']
                image_details_temp['type'] = img_details['meta']['acquisition']['image_type']
                image_details_temp['size_x'] = img_details['meta']['acquisition']['pixelsX']
                image_details_temp['size_y'] = img_details['meta']['acquisition']['pixelsY']
                image_details_temp['age'] = img_details['meta']['clinical']['age_approx']
                image_details_temp['benign_malignant'] = img_details['meta']['clinical']['benign_malignant']
                image_details_temp['diagnosis'] = img_details['meta']['clinical']['diagnosis']
                image_details_temp['diagnosis_confirm_type'] = img_details['meta']['clinical']['diagnosis_confirm_type']
                image_details_temp['melanocytic'] = img_details['meta']['clinical']['melanocytic'] # None, True or False
                image_details_temp['sex'] = img_details['meta']['clinical']['sex']
            except:
                print('[ERROR]: Image don\'t have the looked info (id: %s)' % id)
                continue # Ignore images that don't have alll the looked information

            if save_img(img_details['name'], fetch_image(id)):
                index[id] = image_details_temp
                index_keys = list(index.keys())
                save_index(index)
                downloaded += 1
                if downloaded >= imgs:
                    break

        current_offset += imgs_to_list

        print("%d downloaded imgs (%.2f%%) --- current_offset: %d" % (downloaded, (downloaded/imgs)*100, current_offset) )

def get_dataset_image_types():
    index = load_index()

    types = []

    for k,v in index.items():
        vType = v['type']
        if vType not in types:
            types.append(vType)

    return types

def get_dataset_image_diagnosis():
    index = load_index()

    types = []

    for k,v in index.items():
        vType = v['diagnosis']
        if vType not in types:
            types.append(vType)

    return types

def get_dataset_image_benign_malignant():
    index = load_index()
    types = []

    for k,v in index.items():
        vType = v['benign_malignant']
        if vType not in types:
            types.append(vType)

    return types

def get_dataset_confirmation_types():
    index = load_index()

    types = []

    for k,v in index.items():
        vType = v['diagnosis_confirm_type']
        if vType not in types:
            types.append(vType)

    return types

def get_dataset_ages():
    index = load_index()

    types = []

    for k,v in index.items():
        vType = v['age']
        if vType not in types:
            types.append(vType)

    return types

def get_dataset_sizes():
    index = load_index()

    sizes = []

    for k,v in index.items():
        vSizes = '%s x %s' % (v['size_x'], v['size_y'])
        if vSizes not in sizes:
            sizes.append(vSizes)

    return sizes

def get_melanocytic_options():
    index = load_index()

    types = []

    for k,v in index.items():
        vType = v['melanocytic']
        if vType not in types:
            types.append(vType)

    return types

def get_details_by_name(name):
    index = load_index()

    for v in index.values():
        if v['name'] == name:
            return v

    return None

def get_img_path(img_name):
    return '%s/%s.jpg' % (DATASET_LOCAL_PATH, img_name)

def get_local_dataset_list(filters={}):
    idx = list(load_index().values())
    return list(filter(lambda i: all([i[k] in v for k,v in filters.items()]), idx))

def convert_to_array(img_name, img_width, img_length, rotate_angle=None):
    with Image.open(get_img_path(img_name)) as im:
        im_temp = im.resize((img_width, img_length))
        if rotate_angle:
            im_temp = im_temp.rotate(rotate_angle)

        img_array = np.asarray(im_temp)

        return img_array / 255.0 # Normalize the images to [0, 1]
        # return (img_array-127.5) / 127.5 # Normalize the images to [-1, 1]

def balance_classes(bigger_list, smaller_list):
    '''
    Balance the two classes list, using random techniques
    '''
    total_removed = 0
    total_duplicated = 0
    total_rotated = 0
    while len(bigger_list) > len(smaller_list):
        tech = random()

        if tech <= .5: # 50% of chance to delete random img from bigger_list
            bigger_list.pop(int(random()*len(bigger_list)))
            total_removed += 1
        elif tech <= .7: # 20% of chance to simple duplicate a random img from the smaller_list
            smaller_list.append(choice(smaller_list).copy())
            total_duplicated += 1
        else: # 30% of chance to duplicate a random img from smaller_list with a random rotation
            copying_img = choice(smaller_list).copy()
            total_rotated += 1

            op_temp = random()
            if op_temp <= .33:
                copying_img['rotate'] = 90
            elif op_temp <= .66:
                copying_img['rotate'] = 180
            else:
                copying_img['rotate'] = 270

            smaller_list.append(copying_img)

    logging.info('\nBalacing: %d removed, %d duplicated and %d rotated.' % (total_removed, total_duplicated, total_rotated))

def prepare_classification_index(train_percentage, max_imgs_to_use):
    data = get_local_dataset_list({'type': ['dermoscopic']})
    not_using_data = data[max_imgs_to_use:]
    data = data[:max_imgs_to_use]

    # {'benign': [img_1, img_2, ..., img_n], 'malignant': [img_m, ...]}
    data_by_class = {}
    for d in data:
        temp_group = data_by_class.get(d['benign_malignant'])
        if not temp_group:
            temp_group = []
            data_by_class[d['benign_malignant']] = temp_group
        temp_group.append(d)

    # Balance the classes with random rules
    balance_classes(data_by_class['benign'], data_by_class['malignant'])

    # Shuffle data before split
    shuffle(data_by_class['benign'])
    shuffle(data_by_class['malignant'])

    save_index({'benign': data_by_class['benign'], 'malignant': data_by_class['malignant']},
                DATESET_TRAIN_TEST_INDEX_PATH)

    for d in not_using_data:
        if d['benign_malignant'] == 'benign':
            d['y'] = [1.0, 0.0]
        else:
            d['y'] = [0.0, 1.0]
    save_index(not_using_data, DATASET_NOT_USED_INDEX_PATH)

def prepare_classification_data(train_percentage, img_width, img_length, max_imgs_to_use):
    '''
    This function loads and prepare the image data from database, providing a
    train list and a test list of data
    '''
    def append_img(d):
        return convert_to_array(d['name'], img_width, img_length, d.get('rotate')), d['y']

    prepare_classification_index(train_percentage, max_imgs_to_use)
    data = load_index(DATESET_TRAIN_TEST_INDEX_PATH)

    for d in data['benign']:
        d['y'] = [1.0, 0.0]

    for d in data['malignant']:
        d['y'] = [0.0, 1.0]

    # Split train and test data
    split_idx = int(len(data['benign'])*train_percentage)
    train_data = data['benign'][:split_idx]
    test_data = data['benign'][split_idx:]
    split_idx = int(len(data['malignant'])*train_percentage)
    train_data += data['malignant'][:split_idx]
    test_data += data['malignant'][split_idx:]

    # Mixeup classes
    shuffle(train_data)
    shuffle(test_data)

    # Generate arrays of train and test
    train_X, train_y = [], []
    results = Parallel(n_jobs=4)(delayed(append_img)(d) for d in train_data)
    for r in results:
        train_X.append(r[0])
        train_y.append(r[1])

    test_X, test_y = [], []
    results = Parallel(n_jobs=4)(delayed(append_img)(d) for d in test_data)
    for r in results:
        test_X.append(r[0])
        test_y.append(r[1])

    return np.asarray(train_X), np.asarray(train_y), np.asarray(test_X), np.asarray(test_y)

def prepare_classification_final_data(img_width, img_length):
    def append_img(d):
        return convert_to_array(d['name'], img_width, img_length), d['y']

    data = load_index(DATASET_NOT_USED_INDEX_PATH)

    test_X, test_y = [], []
    results = Parallel(n_jobs=4)(delayed(append_img)(d) for d in data)
    for r in results:
        test_X.append(r[0])
        test_y.append(r[1])

    return np.asarray(test_X), test_y, data

def prepare_gan_data(img_width, img_length, max_imgs_to_use):
    def append_img(d):
        return convert_to_array(d['name'], img_width, img_length)

    data = get_local_dataset_list({'type': ['dermoscopic'], 'benign_malignant': 'malignant'})[5:max_imgs_to_use+5]

    results = Parallel(n_jobs=4)(delayed(append_img)(d) for d in data)
    return np.asarray(results)
