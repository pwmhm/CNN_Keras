# plot dog photos from the dogs vs cats dataset
import matplotlib.pyplot as plt
from matplotlib.image import imread
from random import randint
import csv

id = []
label = []

with open('submission_80_Accuracy.h5.csv', newline='') as csvfile:
	reader = csv.DictReader(csvfile)
	for row in reader:
		id.append(int(row['id']))
		label.append(float(row["label"]))



# define location of dataset
folder = 'Dataset/test/'
# plot first few images
for i in range(64) :
	j = randint(1, 12500)
	idx = int(id.index(j))
	labels = "dog"
	if float(label[idx]) < 0.5 :
		labels = "cat"
	# define subplot
	plt.subplot(8,8,i+1)
	# define filename
	filename = folder + str(j) + '.jpg'
	# load image pixels
	image = imread(filename)
	# plot raw pixel data
	plt.imshow(image/255.0)
	legend = str(j) + "," + labels
	plt.axis("off")
	plt.tight_layout()
	plt.title(labels)
# show the figure
plt.show()