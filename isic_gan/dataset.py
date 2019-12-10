import urllib.request
import json
import os
from pathlib import Path

DATASET_BASE_FOLDER = 'dataset'
DATASET_LOCAL_PATH = 'dataset/raw'
DATASET_LOCAL_INDEX_PATH = 'dataset/raw/index.json'
ISIC_API_URL = 'https://isic-archive.com/api/v1/'

if not Path(DATASET_BASE_FOLDER).exists():
    os.mkdir(DATASET_BASE_FOLDER)
if not Path(DATASET_LOCAL_PATH).exists():
    os.mkdir(DATASET_LOCAL_PATH)

def save_index(content):
    content = json.dumps(content)
    try:
        file = open(DATASET_LOCAL_INDEX_PATH, 'wt')
        file.write(content)
        file.close()
    except:
        print('[ERROR]: Oops, something really wrong happend saving the index file.')

def load_index(filters={}):
    try:
        file = open(DATASET_LOCAL_INDEX_PATH, 'rt')
        content = file.read()
        file.close()
    except OSError:
        content = {}
        save_index(content)
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

        print("%d downloaded imgs (%.2f) --- current_offset: %d" % (downloaded, (downloaded/imgs)*100, current_offset) )

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
