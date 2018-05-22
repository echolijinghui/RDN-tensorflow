from math import log10
from datetime import datetime

from data_tools import *
from model import model

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def train():
    # Prepare Data For Training
    print("\nPrepare Data...")
    train_queue_name='../tf_data/train.tfrecords'
    images_batch, label_batch = decode_from_tfrecords([train_queue_name], image_size, batch_size)

    logits_batch = model(images_batch)  # [batch_size, image_height, image_width, 3]
    #loss = tf.losses.mean_squared_error(logits_batch,label_batch)
    l1_loss = tf.reduce_mean(tf.losses.absolute_difference(label_batch, logits_batch))

    weights_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    regularizer = tf.contrib.layers.l2_regularizer(0.0001)  # return a l2 regularize function
    reg_loss = tf.contrib.layers.apply_regularization(regularizer, weights_list=weights_list)  # apply l2_regularizer,return a scale tensor
    #tf.add_to_collection('losses', reg_loss)
    loss = l1_loss + 0.0001*reg_loss

    # Scalar to keep track for loss
    tf.summary.scalar("l1_loss", l1_loss)
    tf.summary.scalar("loss", loss)

    # # Prepare Data For Validation
    # val_queue_name = '../tf_data/val.tfrecords'
    # val_images_batch, val_label_batch = decode_from_tfrecords([val_queue_name], image_size,  val_batch_size)
    # val_images_batch = tf.reshape(val_images_batch, [val_batch_size, image_size, image_size, c_dim])
    # val_label_batch = tf.reshape(val_label_batch, [val_batch_size, image_size * scale, image_size * scale, c_dim])
    # val_logits_batch = model(val_images_batch)
    # val_mse = tf.reduce_mean(tf.losses.mean_squared_error(val_label_batch, val_logits_batch))

    global_step = tf.Variable(0, name='global_step', trainable=False)
    learning_rate = tf.train.exponential_decay(init_learning_rate, global_step=global_step,
                                               decay_steps=int(lr_update_step), decay_rate=0.5)
    # Create the train_op
    train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss, global_step=global_step)

    tf.summary.scalar("learning_rate", learning_rate)
    merged_summary_op = tf.summary.merge_all()

    with tf.Session() as session:
        summary_writer = tf.summary.FileWriter(log_dir, session.graph)
        saver = tf.train.Saver()
        init = tf.global_variables_initializer()
        session.run(init)
        #tf.local_variables_initializer().run()
        print("\nStarting training...")
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        model_dir = "../model"
        ckpt_path = os.path.join(model_dir, 'model.ckpt')
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        ckpt = tf.train.get_checkpoint_state(model_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(session, ckpt.model_checkpoint_path)
            now_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
            print("\nCheckpoint %s Loading Success, now is in step %s \n" % (ckpt_path, now_step))


        total_loss = 0.0
        total_l1_loss = 0.0
        for step in range(max_iters):
            # try:
            #     while not coord.should_stop():
            _, loss_value, l1_loss_value, lr, summary_str, out_batch, img_batch, gt_batch= session.run(
                [train_op, loss, l1_loss, learning_rate, merged_summary_op, logits_batch, images_batch, label_batch])
            assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

            summary_writer.add_summary(summary_str, step)

            total_loss += loss_value
            total_l1_loss += l1_loss_value
            if step % 100 == 0:
                print('-------------------------------------------')
                print('Currently at step %d of %d.' % (step, max_iters))
                print('%s: average loss on percent training batch is %.5f' % (datetime.now(), total_loss/100.))
                print('%s: average l1_loss on percent training batch is %.5f' % (datetime.now(), total_l1_loss/100.))
                print('%s: present learning rate is %.5f' % (datetime.now(), lr))
                print('\n')
                total_loss = 0.0
                total_l1_loss = 0.0
            if step % 500 == 0:
                x = np.squeeze(img_batch[0])
                y_gt = np.squeeze(gt_batch[0])
                y_out = np.squeeze(out_batch[0])
                cv2.imwrite("%s/%05d_in.png" % (im_dir, step), x)
                cv2.imwrite("%s/%05d_gt.png" % (im_dir, step), y_gt)
                cv2.imwrite("%s/%05d_out.png" % (im_dir, step), y_out)
            if step % 2000 == 0 or (step + 1) == max_iters:
                ckpt_path = os.path.join(model_dir, 'model.ckpt')
                saver.save(session, ckpt_path, global_step=step)
        coord.request_stop()
        coord.join(threads)


log_dir = '../out/logs'  # Name of result logs directory
im_dir = '../out/images' # Name of result images directory during training
model_dir = '../model' # Name of checkpoint directory

max_iters = 600000
init_learning_rate = 0.0001
lr_update_step = 100000

scale = 2 # the size of scale factor for preprocessing input image
c_dim = 3  # The size of image channel
batch_size = 32
val_batch_size = 16
image_size = 48 # The size of LR image input
train()






