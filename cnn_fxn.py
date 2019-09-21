import matplotlib.pyplot as plt
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
import random
import urllib

#Loading 2-layer
cnn = load_model('image_class1.h5')

#Loading InceptionV3
cnn2 = load_model('image_class1_tl.h5')

#Loading 3-layer
cnn3 = load_model('image_class2_3l.h5')

#Loading Xception
cnn3_tl = load_model('image_class2_3l_tl.h5')


def downloader(image_url):
    file_name = random.randrange(1,10000)
    full_file_name = 'Images/Sample_Images/' + str(file_name) + '.jpg'
    urllib.request.urlretrieve(image_url,full_file_name)
    return full_file_name

def image_predictor(path):
    img = load_img(path, target_size=(224, 224))
    plt.imshow(img)
    img = img_to_array(img)
    img = img/255
    img = np.expand_dims(img, axis=0)    
    pred1 = cnn.predict(img)
    pred2 = cnn3.predict(img)
    pred3 = cnn2.predict(img)
    pred4 = cnn3_tl.predict(img)

    return pred1, pred2, pred3, pred4


def predictor_output(path):
    mod1, mod2, mod3, mod4 = image_predictor(path)
    print('VALUE Interpretation:')
    print()
    print('Closer to 1 is Organic, Closer to 0 is Recyclable')
    print()
    print('2-LAYER CNN:')
    print(mod1)
    print()
    print('3-Layer CNN:')
    print(mod2)
    print()
    print('Inception ver3 CNN')
    print(mod3)
    print()
    print('Xception CNN')
    print(mod4)
