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
    ggg = np.array([g,g,g])
    ggg = np.transpose(ggg,(1,2,0))
    ggg = np.expand_dims(ggg, axis=0)
    rgb = np.expand_dims(img, axis=0)
    if is_scale:
        ggg = np.array(ggg).astype(np.float32) / 255.
        rgb = np.array(rgb).astype(np.float32) / 255. 
    return ggg,rgb
