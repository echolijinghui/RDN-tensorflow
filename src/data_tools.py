# -*- coding = utf-8 -*-
import tensorflow as tf
import numpy as np
import time, cv2, random, glob, os


#HR_DIR = '../train_data/train_HR'#''../train_data/hr/'
#val_HR_DIR = '../val_data/val_HR'
#patch_per_img = 20
#patch_size = 48
#batch_size = 2
#instance_perTF = 4000
#train_queue = '../train_data/train.tfrecords'
#val_queue = '../val_data/val.tfrecords'

def dir(dir_name):
    if not os.path.exists(dir_name):
         os.makedirs(dir_name)

def brg2ycrcb(hr_bgr_path):
    BGR = cv2.imread(hr_bgr_path) # [height, width, 3]
    height = BGR.shape[0]
    width = BGR.shape[1]
    YCrCb = cv2.cvtColor(BGR, cv2.COLOR_BGR2YCR_CB)
    #Y_gt = YCrCb[:,:,0]  # [height, width]
    return YCrCb, height, width

# used only in test stage
def modcrop(img, scale =2):
    if len(img.shape) ==3:
        h_i, w_i, _ = img.shape
        h = (h_i / scale) * scale
        w = (w_i / scale) * scale
        img = img[0:h, 0:w, :]
    else:
        h_i, w_i = img.shape
        h = (h_i / scale) * scale
        w = (w_i / scale) * scale
        img = img[0:h, 0:w]
    if h_i != h or w_i !=w:
        print "The image has been resized!"
    return img

def write_img(save_path, img_batch):
    img_s = np.squeeze(img_batch[0]).astype(np.uint8)
    cv2.imwrite(save_path, img_s)

def preprocess(img ,scale = 2):
    input_1 = cv2.resize(img, None, fx =1.0/scale, fy =1.0/scale, interpolation = cv2.INTER_CUBIC)

    flag = random.randint(1, 4)
    if flag == 1:
        kernel_size = (7, 7)
        sigma = 1.6
        input_1 = cv2.GaussianBlur(input_1, kernel_size, sigma)

    return input_1

def data_aug(image, label):
    flag = random.randint(1, 2)
    if flag == 1:
        index = random.randint(-1,1)
        image = cv2.flip(image, index)
        label = cv2.flip(label, index)
    else :
        image, label = image, label
    return image, label

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def encode_to_tfrecords(path, queue_name):
    filenames = sorted(glob.glob(path + '/*'))
    num_files = len(filenames)
    filename = queue_name
    for i in range(num_files):
        img_path = filenames[i]
        print  img_path
        print 'now is %d of %d '%(i+1,num_files)
        #input_gt, height, width = brg2ycrcb(img_path) # [height, width, 3]
        BGR = cv2.imread(img_path)  # [height, width, 3]
        height = BGR.shape[0]
        width = BGR.shape[1]
        for step in range(20):
            writer = tf.python_io.TFRecordWriter(filename)

            in_row_ind = random.randint(0, height - patch_size)
            in_col_ind = random.randint(0, width - patch_size)
            gt_cropped = BGR[in_row_ind:in_row_ind + patch_size,
                                in_col_ind:in_col_ind + patch_size,:]  # [patch_size, patch_size, 3]

            bic_cropped = preprocess(gt_cropped) # [patch_size/2, patch_size/2, 3]

            bic_cropped, gt_cropped = data_aug(bic_cropped, gt_cropped)

            # cv2.imwrite('bic_%d.png' %step, bic_cropped)
            # cv2.imwrite('gt_%d.png' %step, gt_cropped)

            img_raw = bic_cropped.tostring()  # [patch_size/2, patch_size/2]
            label_raw = gt_cropped.tostring() # [patch_size, patch_size]
            example=tf.train.Example(features=tf.train.Features(feature={
                    'label_raw':  _bytes_feature(label_raw),
                    'image_raw': _bytes_feature(img_raw),
                }))
            writer.write(example.SerializeToString())

    writer.close()
    print('Writting End')

def decode_from_tfrecords(filename, patch_size,  batch_size):
    filename_queue = tf.train.string_input_producer(filename)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label_raw' : tf.FixedLenFeature([],tf.string),
                                           'image_raw' : tf.FixedLenFeature([], tf.string)
                                       })

    image = tf.decode_raw(features['image_raw'],tf.uint8)  # LR image
    label = tf.decode_raw(features['label_raw'],tf.uint8)  # HR image

    image = tf.reshape(image,[patch_size,patch_size,3])
    label = tf.reshape(label,[patch_size*2,patch_size*2,3])

    image = tf.cast(image, tf.float32)
    label = tf.cast(label, tf.float32)
    
    min_after_dequeue = 1000
    capacity = min_after_dequeue+3*batch_size
    image,label = tf.train.shuffle_batch([image,label],
                             batch_size=batch_size,
                             num_threads=3,
                             capacity=capacity,
                             min_after_dequeue=min_after_dequeue)
    return image,label

def main():
    # convert training data to tfrecords
    print('convert to train_tfrecords begin')
    start_time = time.time()
    encode_to_tfrecords(HR_DIR, '../train_data/train.tfrecords')
    duration = time.time() - start_time
    print('convert to train_tfrecords end , cost %0.4f sec' %duration)

    # convert validation data to tfrecords
  #  print('convert to val_tfrecords begin')
  #  start_time = time.time()
  #  encode_to_tfrecords(val_HR_DIR, val_queue, 'False')
  #  duration = time.time() - start_time
  #  print('convert to val_tfrecords end , cost %0.4f sec' % duration)

    train_queue=['../train_data/train.tfrecords']
    image, label = decode_from_tfrecords(train_queue,48, batch_size)
    init = tf.global_variables_initializer()
    with tf.Session() as session:
        session.run(init)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        batch_image_np, batch_label_np = session.run([image, label])
        cv2.imwrite('a.jpg',batch_image_np[0])
        cv2.imwrite('b.jpg',batch_label_np[0])

        coord.request_stop()
        coord.join(threads)

#main()
