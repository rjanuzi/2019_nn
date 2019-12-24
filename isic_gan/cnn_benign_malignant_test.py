import dataset as ds

from tensorflow.keras import models

IMGS_WIDTH = 64
IMGS_LENGTH = 64
MAX_IMGS_TO_USE = 256

# MODEL_BKP_NAME = 'benign_malignant_model.h5'
#
train_X, train_y, test_X, test_y = ds.prepare_classification_data(train_percentage=0.8,
                                                    img_width=IMGS_WIDTH,
                                                    img_length=IMGS_LENGTH,
                                                    max_imgs_to_use=MAX_IMGS_TO_USE)
# model = models.load_model(MODEL_BKP_NAME)
#
# predicted_ys = model.predict(test_X)
# print('Real: %s' % test_y)
# print('Predicted: %s' % predicted_ys)

# print(ds.get_dataset_image_benign_malignant())

ys = {}
for t in test_y:
    temp_count = ys.get(str(t))
    if not temp_count:
        temp_count = 0
    temp_count += 1
    ys[str(t)] = temp_count

print(len(train_y))
print(len(test_y))
print(ys)
