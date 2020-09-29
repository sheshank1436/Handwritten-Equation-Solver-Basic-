#!/usr/bin/env python
# coding: utf-8

# ###  Import libraries

# In[2]:


import numpy as np
import cv2
from PIL import Image
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import os
from os import listdir
from os.path import isfile, join
import pandas as pd
from sklearn.utils import shuffle


# ### loading images from the folder
# - first load the images from the respective folders
# -  read the image and the invert it.
# - then convert the image into a binary using the threshold values
# - by using the threshold find the countours
# - using the contours find the edges of the number in the image
# - crop the image with the  respective dimensions 
# - resize the image to 28*28
# - reshape the image and store in an numpy array
# - then append the result into a list

# In[3]:


def load_images_from_folder(folder):
    train_data=[]
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename),cv2.IMREAD_GRAYSCALE)
        img=~img
        if img is not None:
            ret,thresh=cv2.threshold(img,50,255,cv2.THRESH_BINARY)
            #If you pass cv.CHAIN_APPROX_NONE, all the boundary points are stored.
            ctrs,ret=cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
            cnt=sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])
            w=int(28)
            h=int(28)
            maxi=0
            for c in cnt:
                x,y,w,h=cv2.boundingRect(c)
                maxi=max(w*h,maxi)
                if maxi==w*h:
                    x_max=x
                    y_max=y
                    w_max=w
                    h_max=h
            im_crop= thresh[y_max:y_max+h_max+10, x_max:x_max+w_max+10]
            im_resize = cv2.resize(im_crop,(28,28))
            im_resize=np.reshape(im_resize,(784,1))
            train_data.append(im_resize)
    return train_data


# ### Convert the resized images into an array and paste  the output array into a csv file 

# In[4]:



def data_conversion_to_array(dir,dic):
    data=[]
    init=0
    for folder in os.listdir(dir):
        path=os.path.join(dir,folder)
        #print(path)
        label=(path.split("\\")[-1])
        #print([dic[label]])
        if init==0:
            data=load_images_from_folder(path)
            for i in range(0,len(data)):
                data[i]=np.append(data[i],[dic[label]])
            #print(len(data))
        print(folder ,'folder has been converted to array')        
        else:
            data11=load_images_from_folder(path)
            for i in range(0,len(data11)):
                data11[i]=np.append(data11[i],[dic[label]])
            data=np.concatenate((data,data11))
            #print(len(data))
        print(folder ,'folder has been converted to array')    
        init+=1
    df=pd.DataFrame(data,index=None)
    df.to_csv(r'train_final.csv',index=False)    
    


# #### load the data from csv file
# - load the data from csv file using pandas
# - shuffle data using sklearn
# - since the last column let labels=last column and drop it to get the input data

# In[5]:


def load_data_from_csv(file_location):
    df_train=pd.read_csv(file_location,index_col=False)
    df_train= shuffle(df_train)
    labels=df_train[['784']]
    df_train.drop(df_train.columns[[784]],axis=1,inplace=True)
    return df_train,labels

 




