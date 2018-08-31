#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 17 15:14:32 2018

@author: shubhamsinha
"""

import pandas as pd
import numpy as np
import os
import cv2
import random

curdir = os.path.dirname(os.path.realpath(__file__))

def load_data(filename):
    
    data=pd.read_csv(filename)
    labels = np.array(data['emotion'],np.float)
    images=np.array(data['pixels'])
    imagelist=np.array([np.fromstring(image,np.uint8,sep=' ') for image in images])
    num_shape = int(np.sqrt(imagelist.shape[-1]))
    imagelist.shape = (imagelist.shape[0],num_shape,num_shape)
    
    dirs = set(data['Usage'])
    subdirs = set(labels)
    
    class_dir = {}
    for dr in dirs:
        dest = os.path.join(curdir,dr)
        class_dir[dr] = dest
        if not os.path.exists(dest):
            os.mkdir(dest)
        
        
    data = zip(labels,imagelist,data['Usage'])
    
    for d in data:
        destdir = os.path.join(class_dir[d[-1]],str(int(d[0])))
        if not os.path.exists(destdir):
            os.mkdir(destdir)
        img = d[1]
        filepath = unique_name(destdir,d[-1])
        print('[^_^] Write image to %s' % filepath)
        if not filepath:
            continue
        sig = cv2.imwrite(filepath,img)
        if not sig:
            print('Error')
            exit(-1)
            
def unique_name(pardir,prefix,suffix='jpg'):
    filename = '{0}_{1}.{2}'.format(prefix,random.randint(1,10**8),suffix)
    filepath = os.path.join(pardir,filename)
    if not os.path.exists(filepath):
        return filepath
    unique_name(pardir,prefix,suffix)


def main(): 
    filename='fer2013.csv'
    filename=os.path.join(curdir, filename)
    print(filename)
    load_data(filename)
    
    
if __name__ == '__main__':
    main()