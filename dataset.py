from os import listdir
from numpy import asarray
from numpy import save
import tensorflow as tf

path = "Dataset/train/"

images, labels = list(), list()

for file in listdir(path) :
    out_class = 1.0 # Dog
    if file.startswith("cat") :
        out_class = 0.0

    #load image and resize
    #224x224 to simplify, if we use VGG16.
    image = tf.keras.preprocessing.image.load_img((path + file), target_size=(224,224))
    image = tf.keras.preprocessing.image.img_to_array(image)

    images.append(image)
    labels.append(out_class)
images = asarray(images)
labels = asarray(labels)

print(images.shape, labels.shape)

save("Dataset/train/dogs_vs_cats_images_train.npy", images)
save("Dataset/train/dogs_vs_cats_labels_train.npy", labels)

path = "Dataset/test/"

images, labels = list(), list()

for file in listdir(path) :
    id = file.replace(".jpg", "")

    #load image and resize
    #224x224 to simplify, if we use VGG16.
    image = tf.keras.preprocessing.image.load_img((path + file), target_size=(224,224))
    image = tf.keras.preprocessing.image.img_to_array(image)

    images.append(image)
    labels.append(id)
images = asarray(images)
labels = asarray(labels)

print(images.shape, labels.shape)

save("Dataset/test/dogs_vs_cats_images_test.npy", images)
save("Dataset/test/dogs_vs_cats_labels_test.npy", labels)





