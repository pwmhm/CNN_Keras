import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from numpy import load
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

images = load("Dataset/train/dogs_vs_cats_images_train.npy")
labels = load("Dataset/train/dogs_vs_cats_labels_train.npy")

xtv, xtest, ytv, ytest = train_test_split(images,labels, test_size=0.2, shuffle=True)
xtrain, xval, ytrain, yval = train_test_split(xtv,ytv, test_size=0.2, shuffle=True)

learning_rate = 1e-3   #standard learning rate2
epochs        = 100    #massive data
batch_size    = 64
dropout_rate  = 0.5

num_classes = 1

model = Sequential([
 layers.experimental.preprocessing.Rescaling(1./255, input_shape=(224,224,3)),

 layers.Conv2D(16, (3,3), padding = "same", kernel_initializer="he_uniform", activation="relu"),
 layers.MaxPooling2D(2, 2),

 layers.Dropout(0.1),

 layers.Conv2D(32, (3,3), padding="same", kernel_initializer="he_uniform", activation="relu"),
 layers.MaxPooling2D(2, 2),

 layers.Dropout(0.1),

 layers.Conv2D(64, (3, 3), padding="same", kernel_initializer="he_uniform", activation="relu"),
 layers.MaxPooling2D(2, 2),

 layers.Dropout(0.5),

 layers.Conv2D(128, (3, 3), padding="same", kernel_initializer="he_uniform", activation="relu"),
 layers.MaxPooling2D(2, 2),

 layers.Dropout(0.2),

 layers.Flatten(),

 layers.Dense(64, activation="relu", kernel_initializer="he_uniform"),

 layers.Dropout(0.2),

 layers.Dense(num_classes, activation="sigmoid")

])

model.summary()

MCheck = tf.keras.callbacks.ModelCheckpoint('The_Model.h5', monitor="val_accuracy", verbose=1, save_best_only=True)

EDrop = tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=16, mode="max", min_delta=0.0001)

opt = tf.keras.optimizers.SGD(lr=learning_rate, momentum=0.9)

model.compile(optimizer=opt,
              loss="binary_crossentropy",
              metrics=['accuracy'])

history = model.fit(
  x=xtrain,
  y=ytrain,
  shuffle = True,
  validation_data=(xval,yval),
  batch_size=batch_size,
  epochs=epochs,
  callbacks=[MCheck, EDrop]
)

eval_model = tf.keras.models.load_model('The_Model.h5')
evaluation = eval_model.evaluate(xtest,ytest)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()