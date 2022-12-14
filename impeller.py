
# Commented out IPython magic to ensure Python compatibility.
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.image import imread
import cv2
# %matplotlib inline
import warnings
warnings.filterwarnings('ignore')
import holoviews as hv
from holoviews import opts
hv.extension('bokeh')
import json

from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import plot_model
from tensorflow.keras import backend
from sklearn.metrics import confusion_matrix, classification_report
!pip install shap
import shap
from operator import itemgetter



my_data_dir = '/content/drive/MyDrive/casting_data'
train_path = '/content/drive/MyDrive/casting_data/train'
test_path = '/content/drive/MyDrive/casting_data/test'

print(type(my_data_dir))

plt.figure(figsize=(10,8))
ok = plt.imread('/content/drive/MyDrive/casting_data/train/ok_front/cast_ok_0_1.jpeg')
plt.subplot(1, 2, 1)
plt.axis('off')
plt.title("ok", weight='bold', size=20)
plt.imshow(ok,cmap='gray')


ng = plt.imread('/content/drive/MyDrive/casting_data/train/def_front/cast_def_0_2411.jpeg')
plt.subplot(1, 2, 2)
plt.axis('off')
plt.title("def", weight='bold', size=20)
plt.imshow(ng,cmap='gray')

plt.show()

img = cv2.imread('/content/drive/MyDrive/casting_data/train/ok_front/cast_ok_0_1.jpeg')
img_4d = img[np.newaxis]
plt.figure(figsize=(25,10))
generators = {"rotation":ImageDataGenerator(rotation_range=180), 
              "zoom":ImageDataGenerator(zoom_range=0.7), 
              "brightness":ImageDataGenerator(brightness_range=[0.2,1.0]), 
              "height_shift":ImageDataGenerator(height_shift_range=0.7), 
              "width_shift":ImageDataGenerator(width_shift_range=0.7)}

plt.subplot(1, 6, 1)
plt.title("Original", weight='bold', size=15)
plt.imshow(img)
plt.axis('off')
cnt = 2
for param, generator in generators.items():
    image_gen = generator
    gen = image_gen.flow(img_4d, batch_size=1)
    batches = next(gen)
    g_img = batches[0].astype(np.uint8)
    plt.subplot(1, 6, cnt)
    plt.title(param, weight='bold', size=15)
    plt.imshow(g_img)
    plt.axis('off')
    cnt += 1
plt.show()

image_gen = ImageDataGenerator(rescale=1/255, 
                               zoom_range=0.1, 
                               brightness_range=[0.9,1.0])

image_shape = (300,300,1) 
batch_size = 32

train_set = image_gen.flow_from_directory(train_path,
                                            target_size=image_shape[:2],
                                            color_mode="grayscale",
                                            classes={'def_front': 0, 'ok_front': 1},
                                            batch_size=batch_size,
                                            class_mode='binary',
                                            shuffle=True,
                                            seed=0)

test_set = image_gen.flow_from_directory(test_path,
                                           target_size=image_shape[:2],
                                           color_mode="grayscale",
                                           classes={'def_front': 0, 'ok_front': 1},
                                           batch_size=batch_size,
                                           class_mode='binary',
                                           shuffle=False,
                                           seed=0)

train_set.class_indices

backend.clear_session()
model = Sequential()
model.add(Conv2D(filters=16, kernel_size=(7,7), strides=2, input_shape=image_shape, activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
model.add(Conv2D(filters=32, kernel_size=(3,3), strides=1, input_shape=image_shape, activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
model.add(Conv2D(filters=64, kernel_size=(3,3), strides=1, input_shape=image_shape, activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
model.add(Flatten())
model.add(Dense(units=224, activation='relu'))
model.add(Dropout(rate=0.2))
model.add(Dense(units=1, activation='sigmoid'))
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.summary()

#plot_model(model, show_shapes=True, expand_nested=True, dpi=60)

model_save_path = 'casting_product_detection.hdf5'
early_stop = EarlyStopping(monitor='val_loss',patience=2)
checkpoint = ModelCheckpoint(filepath=model_save_path, verbose=1, save_best_only=True, monitor='val_loss')

n_epochs = 9
results = model.fit_generator(train_set, epochs=n_epochs, validation_data=test_set, callbacks=[early_stop,checkpoint])

model_history = results.history
json.dump(model_history, open('model_history.json', 'w'))

losses = pd.DataFrame(model_history)
losses.index = map(lambda x : x+1, losses.index)
losses.head(3)

g = hv.Curve(losses.loss, label='Training Loss') * hv.Curve(losses.val_loss, label='Validation Loss') \
    * hv.Curve(losses.accuracy, label='Training Accuracy') * hv.Curve(losses.val_accuracy, label='Validation Accuracy')
g.opts(opts.Curve(xlabel="Epochs", ylabel="Loss / Accuracy", width=700, height=400,tools=['hover'],show_grid=True,title='Model Evaluation')).opts(legend_position='bottom')

pred_probability = model.predict_generator(test_set)
predictions = pred_probability > 0.5

plt.figure(figsize=(10,6))
plt.title("Confusion Matrix", size=20, weight='bold')
sns.heatmap(
    confusion_matrix(test_set.classes, predictions),
    annot=True,
    annot_kws={'size':14, 'weight':'bold'},
    fmt='d',
    xticklabels=['Defect', 'OK'],
    yticklabels=['Defect', 'OK'])
plt.tick_params(axis='both', labelsize=14)
plt.ylabel('Actual', size=14, weight='bold')
plt.xlabel('Predicted', size=14, weight='bold')
plt.show()

print(classification_report(test_set.classes, predictions, digits=3))

'''

img_pred = Image.open('/content/drive/MyDrive/casting_data/train/def_front/cast_def_0_0.jpeg')
# convert image to numpy array
data = asarray(img_pred)
print(type(data))
# summarize shape
print(data.shape)


image2 = Image.fromarray(data)
print(type(image2))

print(image2.mode)
print(image2.size)

'''

'''
import numpy as np
from PIL import Image
import matplotlib as plt

from numpy import asarray


test_cases = ['ok_front/cast_ok_0_10.jpeg', 'ok_front/cast_ok_0_1026.jpeg', 'ok_front/cast_ok_0_1031.jpeg', 'ok_front/cast_ok_0_1121.jpeg', \
              'ok_front/cast_ok_0_1144.jpeg','def_front/cast_def_0_1059.jpeg', 'def_front/cast_def_0_108.jpeg', 'def_front/cast_def_0_1153.jpeg',\
              'def_front/cast_def_0_1238.jpeg', 'def_front/cast_def_0_1269.jpeg']

plt.figure(figsize=(20,8))
for i in range(len(test_cases)):
    
    img_pred = cv2.imread(test_path + test_cases[i], cv2.IMREAD_GRAYSCALE)
    #img_pred = asarray(img_pred) 
    #img_pred = img_pred/255
    prediction = model.predict(img_pred.reshape(1, *image_shape))
    
    img = cv2.imread(test_path + test_cases[i])
    label = test_cases[i].split("_")[0]
    
    plt.subplot(2, 5, i+1)
    plt.title(f"{test_cases[i].split('/')[1]}\n Actual Label : {label}", weight='bold', size=12)
    if (prediction < 0.5):
        predicted_label = "def"
        prob = (1-prediction.sum()) * 100
    else:
        predicted_label = "ok"
        prob = prediction.sum() * 100
        
    cv2.putText(img=img, text=f"Predicted Label : {predicted_label}", org=(10, 30), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8, color=(255, 0, 255), thickness=2)
    cv2.putText(img=img, text=f"Probability : {'{:.3f}'.format(prob)}%", org=(10, 280), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.7, color=(0, 255, 0), thickness=2)
    plt.imshow(img,cmap='gray')
    plt.axis('off')

plt.show()
'''

#type(img_pred)

"""For OK images testing"""

plt.figure(figsize=(60,45))
img_pred = cv2.imread('/content/drive/MyDrive/casting_data/test/ok_front/cast_ok_0_10.jpeg', cv2.IMREAD_GRAYSCALE)
img_pred = img_pred / 255
prediction = model.predict(img_pred.reshape(1, *image_shape))
  
img = cv2.imread('/content/drive/MyDrive/casting_data/test/ok_front/cast_ok_0_10.jpeg')
label = "Original label: OK"
   
plt.subplot(2, 5, 1)
plt.title(f"{label}", weight='bold', size=28)
if (prediction < 0.5):
    predicted_label = "def"
    prob = (1-prediction.sum()) * 100
else:
    predicted_label = "ok"
    prob = prediction.sum() * 100
        
cv2.putText(img=img, text=f"Predicted Label : {predicted_label}", org=(10, 30))
cv2.putText(img=img, text=f"acc : {'{:.3f}'.format(prob)}%", org=(10, 280))
plt.imshow(img,cmap='gray')
plt.axis('off')

"""For DEF images testing"""

plt.figure(figsize=(60,45))
img_pred = cv2.imread('/content/drive/MyDrive/casting_data/test/def_front/cast_def_0_1134.jpeg', cv2.IMREAD_GRAYSCALE)
img_pred = img_pred / 255
prediction = model.predict(img_pred.reshape(1, *image_shape))
  
img = cv2.imread('/content/drive/MyDrive/casting_data/test/def_front/cast_def_0_1134.jpeg')
label = "Original label: DEF"
   
plt.subplot(2, 5, 1)
plt.title(f"{label}", weight='bold', size=28)
if (prediction < 0.5):
    predicted_label = "def"
    prob = (1-prediction.sum()) * 100
else:
    predicted_label = "ok"
    prob = prediction.sum() * 100
        
cv2.putText(img=img, text=f"Predicted Label : {predicted_label}", org=(10, 30)
cv2.putText(img=img, text=f"Acc : {'{:.3f}'.format(prob)}%", org=(10, 280))
plt.imshow(img,cmap='gray')
plt.axis('off')

def get_image_paths(test_path):

    logger = logging.getLogger(__name__)
    image_extensions = _image_extensions
    dir_contents = list()
    print(get_image_paths)
    return get_image_paths

def get_image_paths(test_path):
    logger = logging.getLogger(__name__)
    image_extensions = _image_extensions
    dir_contents = list()
    print(get_image_paths)
    return get_image_paths
def get_image_paths(test_path):

  logger = logging.getLogger(__name__)
  image_extensions = _image_extensions
  dir_contents = list()
  print(get_image_paths)
  return get_image_paths





















