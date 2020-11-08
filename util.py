import numpy as np
import cv2
import scipy.io as sio

def load_np_image(path, is_scale=True):
    img = cv2.imread(path, -1)
    b,g,r = cv2.split(img)
    diff = g-b
    stain = np.ones(diff.shape,int)
    ones = np.ones(diff.shape,int)
    
    with np.nditer([diff,stain], op_flags=['readwrite']) as it:
        for x,y in it:
            if x >=30 and x<=50:
                y[...]=0
    mask = np.array([stain,ones,ones])
    mask = np.transpose(mask,(1,2,0))
    img = np.multiply(mask,img)
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
    img = np.expand_dims(img, axis=0)
    if is_scale:
        img = np.array(img).astype(np.float32) / 255.
    return img
