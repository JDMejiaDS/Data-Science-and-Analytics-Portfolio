#IMPORTING DEPENDENCIES AND DRIVE
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import imghdr
from google.colab import drive
from google.colab.patches import cv2_imshow
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import load_img
from tensorflow.keras.optimizers import Adam
from keras.preprocessing.image import img_to_array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.metrics import Precision, Recall, BinaryAccuracy
drive.mount('/content/drive')

def get_image_files(carpeta, clase, lista):
  for archivo in os.listdir(os.path.join(carpeta,clase)):
      if archivo.endswith(".jpg") or archivo.endswith(".png") or archivo.endswith(".jpeg"):
          img_ruta = os.path.join(os.path.join(carpeta, clase), archivo)
          img = cv2.imread(img_ruta)
          if img is not None:
              lista.append(img)

def aumentador(origen, clase, destino, estado, aumento):
        for archivo in os.listdir(os.path.join(origen, clase)):
                    img_ruta = os.path.join(os.path.join(origen, clase), archivo)
                    img = cv2.imread(img_ruta)
                    if img is not None:
                        original_img_ruta_aug = os.path.join(os.path.join(destino,estado), archivo)
                        cv2.imwrite(original_img_ruta_aug, img)

                        x = img_to_array(img)
                        x = np.expand_dims(x, axis=0)

                        i = 0
                        for batch in datagen.flow(x, batch_size=1, save_to_dir=os.path.join(destino,estado), save_prefix=clase, save_format='jpg'):
                            i += 1
                            if i >= aumento:
                                break
#RUTES CREATION
ruta_corales = "/content/drive/My Drive/Corales"
blanqueados = 'bleached_corals'
saludables = 'healthy_corals'
ruta_aumentados = "/content/drive/My Drive/Corales_aug"
aug_sal = "bleached_augmented"
aug_blanq = "healthy_augmented"

#SAVING IMAGES TO A LIST
img_blanqueados = []
img_saludables = []

get_image_files(ruta_corales, blanqueados, img_blanqueados)
get_image_files(ruta_corales, saludables, img_saludables)

#GENERATING VISUALIZATION
img_blanqueados[0]
img_saludables[0]

# UPLOADING IMAGES INTO A DATAFRAME LIKE STRUCTURE
ds_not_aug= tf.keras.utils.image_dataset_from_directory(ruta_corales, image_size=(250,250), batch_size=20, shuffle=True)
iter_1 = ds_not_aug.as_numpy_iterator()
bache= iter_1.next()
#DIMENSINALITY VERIFICATION
bache[0].shape
#LABELS VERIFICATION
bache[1]
#IMAGES VISUALIZATION AND IT´S CORRESPONDING LABEL
fig,ax = plt.subplots(ncols=5,figsize=(20,20))
for idx, img in enumerate(bache[0][:5]):
  ax[idx].imshow(img.astype(int))
  ax[idx].title.set_text(bache[1][idx])

#IMAGE NORMALIZATION
ds_NA_escalado = ds_not_aug.map(lambda x,y: (x/255,y))
test_baches=ds_NA_escalado.as_numpy_iterator().next()[0]
test_baches.max()

#CREATING DATA SEGMENTATION INTO 3 SUBSETS TRAINING, VALIDATION AND TESTING
trn = int(len(ds_NA_escalado)*0.7)
val = int(len(ds_NA_escalado)*0.2)+1
tst = int(len(ds_NA_escalado)*0.1)+1
trn_NA = ds_NA_escalado.take(trn)
val_NA = ds_NA_escalado.skip(trn).take(val)
tst_NA = ds_NA_escalado.skip(trn+val).take(tst)

#CREATION OF THE NEURAL NETWORK ARCHITECTURE
red_conv_NA = Sequential()
red_conv_NA.add(Conv2D(30, (3,3), 1, activation='relu', input_shape=(250,250,3)))
red_conv_NA.add(MaxPooling2D())
red_conv_NA.add(Dropout(0.3))
red_conv_NA.add(Conv2D(45, (3,3), 1, activation='relu'))
red_conv_NA.add(MaxPooling2D())
red_conv_NA.add(Dropout(0.2))
red_conv_NA.add(Conv2D(30, (3,3), 1, activation='relu'))
red_conv_NA.add(MaxPooling2D())
red_conv_NA.add(Flatten())
red_conv_NA.add(Dense(250, activation='relu'))
red_conv_NA.add(Dropout(0.5))
red_conv_NA.add(Dense(1, activation='sigmoid'))

#MODEL COMPILATION
red_conv_NA.compile(optimizer=Adam(learning_rate=0.0015), loss='binary_crossentropy', metrics=['accuracy'])
#LAYERS STRUCTURE AND NUMBER OF PARAMETERS
red_conv_NA.summary()
#MODEL TRAINING
historial_NA = red_conv_NA.fit(trn_NA, epochs=20, validation_data=val_NA)
#LOSS CURVE VISUALIZATION
fig = plt.figure()
plt.plot(historial_NA.history['loss'], color='purple', label='loss', ls = "--")
plt.plot(historial_NA.history['val_loss'], color='teal', label='val_loss', ls = ":")
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss Curve')
plt.legend()
plt.show()
#ACCURACY VISUALIZATION
fig = plt.figure()
plt.plot(historial_NA.history['accuracy'], color='purple', label='accuracy', ls = "--")
plt.plot(historial_NA.history['val_accuracy'], color='teal', label='val_accuracy', ls = ":")
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Accuracy Curve')
plt.legend()
plt.show()
#COMPUTATION OF PERFORMANCE METRICS AND GENERATION OF PREDICTIONS
Prec_NA= Precision()
Rec_NA = Recall()
Acc_NA = BinaryAccuracy()
for bache in tst_NA.as_numpy_iterator():
  X, y = bache[0], bache[1]
  y_pred = red_conv_NA.predict(X)
  Prec_NA.update_state(y, y_pred)
  Rec_NA.update_state(y, y_pred)
  Acc_NA.update_state(y, y_pred)

#VALUES PER EACH METRIC
print(Prec_NA.result().numpy(), Rec_NA.result().numpy(), Acc_NA.result().numpy())

#PARAMETERS GENERATION FOR DATA AUGMENTATION
datagen = ImageDataGenerator(
    rescale=1.0/255.0,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

#GENERATING DATA AUGMENTATION
aumentador(ruta_corales, saludables, ruta_aumentados, aug_sal, 1)
aumentador(ruta_corales, blanqueados, ruta_aumentados, aug_blanq, 1)

#VISUALIZATION OF THE NUMBER OF AUGMENTED IMAGES IN HEALTHY CORALS
num_aug_sal = len([f for f in os.listdir(os.path.join(ruta_aumentados, aug_sal))])
num_arch_sal = len([f for f in os.listdir(os.path.join(ruta_corales,saludables))])
print(f"number of agumented images:{num_aug_sal}, number of Not augmented images:{num_arch_sal}")

#VISUALIZATION OF THE NUMBER OF AUGMENTED IMAGES IN BLEACHED CORALS
num_aug_bl = len([f for f in os.listdir(os.path.join(ruta_aumentados, aug_blanq))])
num_arch_bl = len([f for f in os.listdir(os.path.join(ruta_corales,blanqueados))])
print(f"number of agumented images:{num_aug_bl}, number of Not augmented images:{num_arch_bl}")

#VISUALIZATION OF THE NUMBER OF AUGMENTED IMAGES IN CORALS
ds = tf.keras.utils.image_dataset_from_directory(ruta_aumentados, image_size=(250,250), batch_size=20, shuffle=True)
iter = ds.as_numpy_iterator()
bache= iter.next()

#VISUALIZATION OF AUGMENTED DATA
fig,ax = plt.subplots(ncols=5,figsize=(20,20))
for idx, img in enumerate(bache[0][:5]):
  ax[idx].imshow(img.astype(int))
  ax[idx].title.set_text(bache[1][idx])

ds_escalado = ds.map(lambda x,y: (x/255,y))
test_baches=ds_escalado.as_numpy_iterator().next()[0]
test_baches.max()

trn = int(len(ds_escalado)*0.7)
val = int(len(ds_escalado)*0.2)+1
tst = int(len(ds_escalado)*0.1)+1
trn_ds = ds_escalado.take(trn)
val_ds = ds_escalado.skip(trn).take(val)
tst_ds = ds_escalado.skip(trn+val).take(tst)

#augmented data CNN architecture
red_conv = Sequential()
red_conv.add(Conv2D(30, (3,3), 1, activation='relu', input_shape=(250,250,3)))
red_conv.add(MaxPooling2D())
red_conv.add(Dropout(0.3))
red_conv.add(Conv2D(45, (3,3), 1, activation='relu'))
red_conv.add(MaxPooling2D())
red_conv.add(Dropout(0.2))
red_conv.add(Conv2D(30, (3,3), 1, activation='relu'))
red_conv.add(MaxPooling2D())
red_conv.add(Flatten())
red_conv.add(Dense(250, activation='relu'))
red_conv.add(Dropout(0.5))
red_conv.add(Dense(1, activation='sigmoid'))
red_conv.compile(optimizer=Adam(learning_rate=0.0015), loss='binary_crossentropy', metrics=['accuracy'])

#augmented data CNN traning
historial = red_conv.fit(trn_ds, epochs=20, validation_data=val_ds)

#Performance metrics computation
Prec= Precision()
Rec = Recall()
Acc = BinaryAccuracy()
for bache in tst_ds.as_numpy_iterator():
  X, y = bache[0], bache[1]
  y_pred = red_conv.predict(X)
  Prec.update_state(y, y_pred)
  Rec.update_state(y, y_pred)
  Acc.update_state(y, y_pred)

print(Prec.result().numpy(), Rec.result().numpy(), Acc.result().numpy())

#Saving Model
red_conv_NA.save("Bleached_corals_detector.h5") 