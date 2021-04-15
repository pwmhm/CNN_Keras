import tensorflow as tf
from numpy import load
import csv
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import matplotlib.pyplot as plt
from random import randint

image = load("Dataset/test/dogs_vs_cats_images_test.npy")
label = load("Dataset/test/dogs_vs_cats_labels_test.npy")
model_name = "The_Model.h5"

print(len(label))


model = tf.keras.models.load_model(model_name)

model.summary()

results = model.predict(image)

plt.figure(figsize=(10, 10))
for i in range(48) :
    j = randint(0,len(image))
    labs = "doge"
    if float(results[j]) < 0.5 :
        labs = "catto"
    plt.subplot(7, 7, i+1)
    plt.imshow(image[j]/255.0)
    plt.title(labs)
plt.show()


####for checking log loss in kaggle
str_csv = "submission_" + model_name + ".csv"

with open(str_csv, 'w', newline='') as file:
    fieldnames = ['id', 'label']
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()

    for i in range(len(results)) :
        writer.writerow({'id': label[i], 'label': str(results[i])[1:-1]})
