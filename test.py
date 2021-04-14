import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import matplotlib.pyplot as plt

# learning_rate = 1e-5   #standard learning rate2
# epochs        = 11     #massive data
# batch_size    = 64
# dropout_rate  = 0.5

test_dataset = tf.data.experimental.load("processed_dataset/test/", (tf.TensorSpec(shape=(300, 300, 3), dtype=tf.float32, name=None),
 tf.TensorSpec(shape=(), dtype=tf.int64, name=None)), compression="GZIP")

test_dataset = test_dataset.batch(64)

model = tf.keras.models.load_model("The_Model.h5")

results = model.evaluate(test_dataset)