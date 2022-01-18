import os
import cv2

direc = 'a'
count=1000
#cv2.imwrite("a/"+"A-%d.jpg" %count,f1)
for img in os.listdir(direc):
    f1 = cv2.imread('a/'+img)
    if(img[0]=='A'):
    	cv2.imwrite("symbols/"+"A-%d.jpg" %count,f1)
    	count = count+1
    if(img[0]=='B'):
    	cv2.imwrite("symbols/"+"B-%d.jpg" %count,f1)
    	count = count+1
    if(img[0]=='C'):
    	cv2.imwrite("symbols/"+"C-%d.jpg" %count,f1)
    	count = count+1
    if(img[0]=='D'):
    	cv2.imwrite("symbols/"+"D-%d.jpg" %count,f1)
    	count = count+1
    if(img[0]=='E'):
    	cv2.imwrite("symbols/"+"E-%d.jpg" %count,f1)
    	count = count+1
    if(img[0]=='F'):
    	cv2.imwrite("symbols/"+"F-%d.jpg" %count,f1)
    	count = count+1
    if(img[0]=='G'):
    	cv2.imwrite("symbols/"+"G-%d.jpg" %count,f1)
    	count = count+1
    if(img[0]=='H'):
    	cv2.imwrite("symbols/"+"H-%d.jpg" %count,f1)
    	count = count+1
    if(img[0]=='I'):
    	cv2.imwrite("symbols/"+"I-%d.jpg" %count,f1)
    	count = count+1
    if(img[0]=='K'):
    	cv2.imwrite("symbols/"+"K-%d.jpg" %count,f1)
    	count = count+1
    if(img[0]=='L'):
    	cv2.imwrite("symbols/"+"L-%d.jpg" %count,f1)
    	count = count+1
    if(img[0]=='M'):
    	cv2.imwrite("symbols/"+"M-%d.jpg" %count,f1)
    	count = count+1
    if(img[0]=='N'):
    	cv2.imwrite("symbols/"+"N-%d.jpg" %count,f1)
    	count = count+1
    if(img[0]=='O'):
    	cv2.imwrite("symbols/"+"O-%d.jpg" %count,f1)
    	count = count+1
    if(img[0]=='P'):
    	cv2.imwrite("symbols/"+"P-%d.jpg" %count,f1)
    	count = count+1
    if(img[0]=='Q'):
    	cv2.imwrite("symbols/"+"Q-%d.jpg" %count,f1)
    	count = count+1
    if(img[0]=='R'):
    	cv2.imwrite("symbols/"+"R-%d.jpg" %count,f1)
    	count = count+1
    if(img[0]=='S'):
    	cv2.imwrite("symbols/"+"S-%d.jpg" %count,f1)
    	count = count+1
    if(img[0]=='T'):
    	cv2.imwrite("symbols/"+"T-%d.jpg" %count,f1)
    	count = count+1
    if(img[0]=='U'):
    	cv2.imwrite("symbols/"+"U-%d.jpg" %count,f1)
    	count = count+1
    if(img[0]=='V'):
    	cv2.imwrite("symbols/"+"V-%d.jpg" %count,f1)
    	count = count+1
    if(img[0]=='W'):
    	cv2.imwrite("symbols/"+"W-%d.jpg" %count,f1)
    	count = count+1
    if(img[0]=='X'):
    	cv2.imwrite("symbols/"+"X-%d.jpg" %count,f1)
    	count = count+1
    if(img[0]=='Y'):
    	cv2.imwrite("symbols/"+"Y-%d.jpg" %count,f1)
    	count = count+1  
