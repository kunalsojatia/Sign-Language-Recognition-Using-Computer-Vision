from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import imutils
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
	help="path to trained model model")
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])
orig = image.copy()
 
# pre-process the image for classification
image = cv2.resize(image, (256, 256))
image = image.astype("float") / 255.0
image = img_to_array(image)
image = np.expand_dims(image, axis=0)

print("[INFO] loading network...")
model = load_model(args["model"])

arr = model.predict(image)[0]
print(model.predict(image))

index = np.argmax(arr)

if index==0:
   label="A"
if index==1:
   label="B"
if index==2:
   label="C"
if index==3:
   label="D"
if index==4:
   label="E"
if index==5:
   label="F"
if index==6:
   label="G"
if index==7:
   label="H"
if index==8:
   label="I"
if index==9:
   label="K"
if index==10:
   label="L"
if index==11:
   label="M"
if index==12:
   label="N"
if index==13:
   label="O"
if index==14:
   label="P"
if index==15:
   label="Q"
if index==16:
   label="R"
if index==17:
   label="S"
if index==18:
   label="T"
if index==19:
   label="U"
if index==20:
   label="V"
if index==21:
   label="W"
if index==22:
   label="X"
if index==23:
   label="Y"
label = "{}: {:.2f}%".format(label, arr[index] * 100)

output = imutils.resize(orig, width=400)
cv2.putText(output, label, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,
	0.7, (0, 255, 0), 2)
 
# show the output image
cv2.imshow("Output", output)
cv2.waitKey(0)
