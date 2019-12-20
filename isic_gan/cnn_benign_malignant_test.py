import dataset as ds

from tensorflow.keras import models

IMGS_WIDTH = 256
IMGS_LENGTH = 256
MAX_IMGS_TO_USE = 200

MODEL_BKP_NAME = 'benign_malignant_model.h5'

_, _, test_X, test_y = ds.prepare_classification_data(train_percentage=0.8,
                                                    img_width=IMGS_WIDTH,
                                                    img_length=IMGS_LENGTH,
                                                    max_imgs_to_use=MAX_IMGS_TO_USE,
                                                    classify_benign_malignant=True)

model = models.load_model(MODEL_BKP_NAME)

predicted_ys = model.predict(test_X)
print('Real: %s' % test_y)
print('Predicted: %s' % predicted_ys)
