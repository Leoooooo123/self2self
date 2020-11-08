import numpy as np
import cv2
import scipy.io as sio


def add_gaussian_noise(img, model_path, sigma):
    index = model_path.rfind("/")
    if sigma > 0:
        noise = np.random.normal(scale=sigma / 255., size=img.shape).astype(np.float32)
        sio.savemat(model_path[0:index] + '/noise.mat', {'noise': noise})
        noisy_img = (img + noise).astype(np.float32)
    else:
        noisy_img = img.astype(np.float32)
    cv2.imwrite(model_path[0:index] + '/noisy.png',
                np.squeeze(np.int32(np.clip(noisy_img, 0, 1) * 255.)))
    return noisy_img


def load_np_image(path, is_scale=True):
    img = cv2.imread(path, -1)
    b,g,r = cv2.split(img)
    if b.ndim == 2:
        b = np.expand_dims(b, axis=2)
        g = np.expand_dims(g, axis=2)
        r = np.expand_dims(r, axis=2)
    b = np.expand_dims(b, axis=0)
    g = np.expand_dims(g, axis=0)
    r = np.expand_dims(r, axis=0)
    if is_scale:
        b = np.array(b).astype(np.float32) / 255.
        g = np.array(g).astype(np.float32) / 255. 
        r = np.array(r).astype(np.float32) / 255. 
    return b,g,r
