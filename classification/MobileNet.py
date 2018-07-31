from keras.applications.mobilenet import MobileNet
from keras.preprocessing import image
from keras.applications.mobilenet import preprocess_input, decode_predictions
import numpy as np
import os
import tensorflow as tf
from PIL import Image

from keras.backend.tensorflow_backend import set_session

os.environ['CUDA_VISIBLE_DEVICES'] = '2'

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.45
set_session(tf.Session(config=config))

model = MobileNet(weights='imagenet')

img_folder = './test_resnet50/DIV2K_X4/'
img_files = os.listdir(img_folder)

resize_img_folder = './test_mobilenet/resize_image/'

for img_path in img_files:
    img = image.load_img(img_folder + img_path, target_size=(224, 224), interpolation='lanczos')
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    preds = model.predict(x)
    # decode the results into a list of tuples (class, description, probability)
    # (one such list for each sample in the batch)

    predic_origin = decode_predictions(preds, top=3)[0][0]

    img = image.load_img(img_folder + img_path, target_size=(160, 160), interpolation='lanczos')
    img = img.resize((224, 224), resample=1)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    preds = model.predict(x)
    # decode the results into a list of tuples (class, description, probability)
    # (one such list for each sample in the batch)

    predic_resize = decode_predictions(preds, top=3)[0][0]

    if predic_resize[0] != predic_origin[0] and predic_origin[2] > 0.5:
        #img.save(resize_img_folder + img_path)
        print(img_path, 'origin: ', predic_origin, ' resize: ', predic_resize)

    # Predicted: [(u'n02504013', u'Indian_elephant', 0.82658225), (u'n01871265', u'tusker', 0.1122357), (u'n02504458', u'African_elephant', 0.061040461)]

'''
img = image.load_img('./test_resnet50/DIV2K/0870.png', target_size=(150, 150))
img = img.resize((224, 224))
img.save('0870_resize.png')
'''


'''
model = MobileNet(weights='imagenet')

img = image.load_img('./test_resnet50/DIV2K/0870.png', target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

preds = model.predict(x)
# decode the results into a list of tuples (class, description, probability)
# (one such list for each sample in the batch)
print('Predicted:', decode_predictions(preds, top=3)[0])
# Predicted: [(u'n02504013', u'Indian_elephant', 0.82658225), (u'n01871265', u'tusker', 0.1122357), (u'n02504458', u'African_elephant', 0.061040461)]

img = image.load_img('./test_resnet50/DIV2K/0870.png', target_size=(150, 150))
img = img.resize((224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

preds = model.predict(x)
# decode the results into a list of tuples (class, description, probability)
# (one such list for each sample in the batch)
print('Predicted:', decode_predictions(preds, top=3)[0])
# Predicted: [(u'n02504013', u'Indian_elephant', 0.82658225), (u'n01871265', u'tusker', 0.1122357), (u'n02504458', u'African_elephant', 0.061040461)]

'''