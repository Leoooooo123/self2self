import tensorflow as tf
import network.Punet
import numpy as np

import util
import cv2
import os

TF_DATA_TYPE = tf.float32
LEARNING_RATE = 1e-4
N_PREDICTION = 100
N_SAVE = 1000
N_STEP = 50000


def train(file_path, dropout_rate, is_realnoisy=False):
    print(file_path)
    tf.reset_default_graph()
    img = util.load_np_image(file_path)
    _, w, h, c = np.shape(img)
    model_path = file_path[0:file_path.rfind(".")] + "/" + "/model/Self2Self/"
    os.makedirs(model_path, exist_ok=True)
    model = network.Punet.build_denoising_unet(img, dropout_rate, is_realnoisy)

    loss = model['training_error']
    summay = model['summary']
    saver = model['saver']
    our_image = model['our_image']
    is_flip_lr = model['is_flip_lr']
    is_flip_ud = model['is_flip_ud']
    avg_op = model['avg_op']
    slice_avg = model['slice_avg']
    optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)

    avg_loss = 0
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        summary_writer = tf.summary.FileWriter(model_path, sess.graph)
        for step in range(N_STEP):
            feet_dict = {is_flip_lr: np.random.randint(0, 2), is_flip_ud: np.random.randint(0, 2)}
            _, _op, loss_value, merged, o_image = sess.run([optimizer, avg_op, loss, summay, our_image],
                                                           feed_dict=feet_dict)
            avg_loss += loss_value
            if (step + 1) % N_SAVE == 0:

                print("After %d training step(s)" % (step + 1),
                      "loss  is {:.9f}".format(avg_loss / N_SAVE))
                avg_loss = 0
                sum = np.float32(np.zeros(our_image.shape.as_list()))
                for j in range(N_PREDICTION):
                    feet_dict = {is_flip_lr: np.random.randint(0, 2), is_flip_ud: np.random.randint(0, 2)}
                    o_avg, o_image = sess.run([slice_avg, our_image], feed_dict=feet_dict)
                    sum += o_image
                o_image = np.squeeze(np.uint8(np.clip(sum / N_PREDICTION, 0, 1) * 255))
                o_avg = np.squeeze(np.uint8(np.clip(o_avg, 0, 1) * 255))
                if is_realnoisy:
                    cv2.imwrite(model_path + 'Self2Self-' + str(step + 1) + '.png', o_avg)
                else:
                    cv2.imwrite(model_path + 'Self2Self-' + str(step + 1) + '.png', o_image)
                saver.save(sess, model_path + "model.ckpt-" + str(step + 1))

            summary_writer.add_summary(merged, step)


if __name__ == '__main__':
    path = './testsets/stain.png'
    train(path,0.7)
