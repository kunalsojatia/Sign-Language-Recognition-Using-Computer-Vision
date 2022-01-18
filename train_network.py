import matplotlib
matplotlib.use("Agg")
from keras.utils import plot_model
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from keras import utils as np_utils
from lenet import LeNet
from imutils import paths
import pydot
import graphviz
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import argparse 
import random
import cv2
import os

#argument parse
ap = argparse.ArgumentParser()
ap.add_argument("-d","--dataset",required=True,help="path to input dataset")
ap.add_argument("-m","--model",required=True,help="path to output model")
ap.add_argument("-p","--plot",type=str,default="plot.png",help="path to output accuracy/loss plot")
args = vars(ap.parse_args())

EPOCHS = 20
INIT_LR = 1e-3
BS = 10

print("[INFO] loading images...")
data = []
labels = []


imagePaths = sorted(list(paths.list_images(args["dataset"])))
random.seed(42)
random.shuffle(imagePaths)
label_=0
for imagePath in imagePaths:
    image = Image.open(imagePath)
    new_width  = 256
    new_height = 256
    image = image.resize((new_width, new_height), Image.ANTIALIAS) 
    image.load()
    image = np.asarray(image,dtype="int32")
    data.append(image/255.0)
       
    label = imagePath.split(os.path.sep)
    if "A" in label[1]:
        label_ = 0
    if "B" in label[1]:
        label_ = 1 
    if "C" in label[1]:
        label_ = 2
    if "D" in label[1]:
        label_ = 3
    if "E" in label[1]:
        label_ = 4
    if "F" in label[1]:
        label_ = 5 
    if "G" in label[1]:
        label_ = 6
    if "H" in label[1]:
        label_ = 7
    if "I" in label[1]:
        label_ = 8
    if "K" in label[1]:
        label_ = 9 
    if "L" in label[1]:
        label_ = 10
    if "M" in label[1]:
        label_ = 11
    if "N" in label[1]:
        label_ = 12
    if "O" in label[1]:
        label_ = 13 
    if "P" in label[1]:
        label_ = 14
    if "Q" in label[1]:
        label_ = 15
    if "R" in label[1]:
        label_ = 16
    if "S" in label[1]:
        label_ = 17 
    if "T" in label[1]:
        label_ = 18
    if "U" in label[1]:
        label_ = 19
    if "V" in label[1]:
        label_ = 20
    if "W" in label[1]:
        label_ = 21 
    if "X" in label[1]:
        label_ = 22
    if "Y" in label[1]:
        label_ = 23
    labels.append(label_)
data = np.array(data)
labels = np.array(labels)

(trainX,testX,trainY,testY) = train_test_split(data,labels,test_size=0.25,random_state=42)

trainY = np_utils.to_categorical(trainY,num_classes=24)
testY = np_utils.to_categorical(testY,num_classes=24)

#initialize the model
print("[INFO] compiling model..")
model = LeNet.build(width=256,height=256,depth=3,classes=24)
opt = Adam(lr=INIT_LR,decay=INIT_LR/EPOCHS)
model.compile(loss="categorical_crossentropy",optimizer=opt,metrics=["accuracy"])

#train the network
print("[INFO] training network..")
H = model.fit(trainX,trainY,
          epochs=25,
          batch_size= 25)

score = model.evaluate(testX, testY, batch_size=18)
plot_model(model, to_file='model.png')
print(score)

print("[INFO] serializing network")
model.save(args["model"])
model.summary()

# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
N = 25
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(args["plot"])
