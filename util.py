import numpy as np
import cv2
import scipy.io as sio

def load_np_image(path, is_scale=True):
    img = cv2.imread(path, -1)
    b,g,r = cv2.split(img)
    diff = g-b
    stain = np.zeros(diff.shape,int)
    with np.nditer([diff,stain], op_flags=['readwrite']) as it:
        for x,y in it:
            if x >=30 and x<=50:
                y[...]=1
    if b.ndim == 2:
        b = np.expand_dims(b, axis=2)
        g = np.expand_dims(g, axis=2)
        r = np.expand_dims(r, axis=2)
        stain = np.expand_dims(stain, axis=2)
    b = np.expand_dims(b, axis=0)
    g = np.expand_dims(g, axis=0)
    r = np.expand_dims(r, axis=0)
    stain = np.expand_dims(stain, axis=0)
    if is_scale:
        b = np.array(b).astype(np.float32) / 255.
        g = np.array(g).astype(np.float32) / 255.
        r = np.array(r).astype(np.float32) / 255.
    # print(b.shape)
    return b,g,r,stain
