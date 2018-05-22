from datetime import datetime
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.ops import control_flow_ops
import os, cv2, glob
import random

from math import log10
from model import model


os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def myeval(img,label_path):
          images = tf.placeholder(tf.float32,shape=[1,None,None,3])
          logits_eval = model(images)

          saver = tf.train.Saver(tf.global_variables())
          init = tf.global_variables_initializer()

          sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
          sess.run(init)

          tf.train.start_queue_runners(sess=sess)
         
          model_file=tf.train.latest_checkpoint('../model')
          saver.restore(sess,model_file)

          logits_image = sess.run(logits_eval,feed_dict={images:img})
          logits_image = np.squeeze(logits_image).astype(np.uint8)
          print label_path.replace('hr','out')
          cv2.imwrite(label_path.replace('hr','out'),logits_image)

path = '../../qust_img/hr'
label_path = sorted(glob.glob(path + "/*"))
num_files = len(label_path)
for i in range(num_files):
  label = cv2.imread(label_path[i])
  h,w,_=label.shape
  img = cv2.resize(label,(w/2,h/2),interpolation=cv2.INTER_CUBIC)#.reshape(1,h/2,w/2,3)
  img = img.reshape(1,h/2,w/2,3)
  img = np.array(img)
  myeval(img,label_path[i])
  '''

  '''

