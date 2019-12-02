import urllib.request
import json

DATASET_LOCAL_PATH = 'dataset/raw'
DATASET_LOCAL_INDEX_PATH = 'dataset/raw/index.json'
ISIC_API_URL = 'https://isic-archive.com/api/v1/'

def saveIndex(content):
    content = json.dumps(content)
    try:
        file = open(DATASET_LOCAL_INDEX_PATH, 'wt')
        file.write(content)
        file.close()
    except:
        print('[ERROR]: Oops, something really wrong happend saving the index file.')

def loadIndex():
    try:
        file = open(DATASET_LOCAL_INDEX_PATH, 'rt')
        content = file.read()
        file.close()
    except OSError:
        content = {}
        saveIndex(content)
        content = '{}'
    finally:
        return json.loads(content)

def getImageList(imgs_to_list=100, offset=0):
    try:
        url = urllib.request.urlopen(ISIC_API_URL+'image?limit=%d&offset=%d' % (imgs_to_list, offset))
        return json.loads(url.read().decode())
    except:
        print('[ERROR]: Oops, something really bad happend at downloading the image list with offset %d' % offset)
        return []

def getImageDetails(id):
    try:
        url = urllib.request.urlopen(ISIC_API_URL+'image/%s' % id)
        return json.loads(url.read().decode())
    except:
        print('[ERROR]: Oops, something really bad happend at downloading the image details with id %s' % id)
        return None

def getImage(id):
    try:
        url =  urllib.request.urlopen(ISIC_API_URL+'image/%s/download?contentDisposition=inline' % id)
        return url.read()
    except:
        print('[ERROR]: Oops, something really bad happend at downloading the image with id %s' % id)
        return b''

def saveImg(fileName, data):
    try:
        file = open('%s/%s.jpg' % (DATASET_LOCAL_PATH, fileName), "wb")
        file.write(data)
        file.close()
        return True
    except:
        print('[ERROR]: Oops, something really bad happend at saving a image')
        return False

def downloadImgs(imgs=100, start_offset=0):
    index = loadIndex()
    index_keys = list(index.keys())
    downloaded = 0
    current_offset = start_offset
    imgs_to_list = 200

    while True:

        if downloaded >= imgs:
            break

        for img_ref in getImageList(imgs_to_list, current_offset):

            id = img_ref['_id']

            if id in index_keys:
                continue # Ignore images that are already at index

            img_details = getImageDetails(id)
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

            if saveImg(img_details['name'], getImage(id)):
                index[id] = image_details_temp
                index_keys = list(index.keys())
                saveIndex(index)
                downloaded += 1
                if downloaded >= imgs:
                    break

        current_offset += imgs_to_list

        print("%d downloaded imgs (%.2f) --- current_offset: %d" % (downloaded, (downloaded/imgs)*100, current_offset) )

def getDatasetImageTypes():
    index = loadIndex()

    types = []

    for k,v in index.items():
        vType = v['type']
        if vType not in types:
            types.append(vType)

    return types

def getDatasetImageDiagnosis():
    index = loadIndex()

    types = []

    for k,v in index.items():
        vType = v['diagnosis']
        if vType not in types:
            types.append(vType)

    return types

def getDatasetImageBenignMalignant():
    index = loadIndex()
    types = []

    for k,v in index.items():
        vType = v['benign_malignant']
        if vType not in types:
            types.append(vType)

    return types

def getDatasetConfirmationTypes():
    index = loadIndex()

    types = []

    for k,v in index.items():
        vType = v['diagnosis_confirm_type']
        if vType not in types:
            types.append(vType)

    return types

def getDatasetAges():
    index = loadIndex()

    types = []

    for k,v in index.items():
        vType = v['age']
        if vType not in types:
            types.append(vType)

    return types

def getDatasetSizes():
    index = loadIndex()

    sizes = []

    for k,v in index.items():
        vSizes = '%s_%s' % (v['size_x'], v['size_y'])
        if vSizes not in sizes:
            sizes.append(vSizes)

    return sizes

def getMelanocyticOptions():
    index = loadIndex()

    types = []

    for k,v in index.items():
        vType = v['melanocytic']
        if vType not in types:
            types.append(vType)

    return types

def getDetailsByName(name):
    index = loadIndex()

    for v in index.values():
        if v['name'] == name:
            return v

    return None

def getImgPath(imgName):
    return '%s/%s.jpg' % (DATASET_LOCAL_PATH, imgName)

def loadImage(imgName):
    file = open(getImgPath(imgName), 'rb')
    content = file.read()
    file.close()
    return content

def saveTrainData(train_ins, train_outs, labels):
    obj = {'_in':train_ins, '_out':train_outs, 'labels':labels}
    file = open('dataset/traindata.json', 'wt')
    file.write(json.dumps(obj))
    file.close()

def loadTrainData():
    file = open('dataset/traindata.json', 'rt')
    content = file.read()
    file.close()
    loaded = json.loads(content)
    return loaded['_in'], loaded['_out'], loaded['labels']

def saveTestData(test_ins, test_outs):
    obj = {'_in':test_ins, '_out':test_outs}
    file = open('dataset/testdata.json', 'wt')
    file.write(json.dumps(obj))
    file.close()

def loadTestData():
    file = open('dataset/testdata.json', 'rt')
    content = file.read()
    file.close()
    loaded = json.loads(content)
    return loaded['_in'], loaded['_out']
