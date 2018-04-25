import tensorflow as tf
import numpy as np

scale = 2
D = 16
C = 8
G = 64
G0 = 64
ks = 3
c_dim = 3

weightsD = {
    'w_D_1': tf.Variable(tf.random_normal([1, 1, G * D, G0], stddev=np.sqrt(2.0/1/(G * D))), name='w_D_1'),
    'w_D_2': tf.Variable(tf.random_normal([ks, ks, G0, G0], stddev=np.sqrt(2.0/ks**2/G0)), name='w_D_2') }
biasesD = {
    'b_D_1': tf.Variable(tf.zeros([G0], name='b_D_1')),
    'b_D_2': tf.Variable(tf.zeros([G0], name='b_D_2')) }

weightsR = {}
biasesR = {}
for i in range(1, D+1):
    for j in range(1, C+1):
        weightsR.update({'w_R_%d_%d' % (i, j): tf.Variable(tf.random_normal([ks, ks, G * j, G], stddev=np.sqrt(2.0/ks**2/(G * j))), name='w_R_%d_%d' % (i, j))})
        biasesR.update({'b_R_%d_%d' % (i, j): tf.Variable(tf.zeros([G], name='b_R_%d_%d' % (i, j)))})
    weightsR.update({'w_R_%d_%d' % (i, C+1): tf.Variable(tf.random_normal([1, 1, G * (C+1), G], stddev=np.sqrt(2.0/1/(G * (C+1)))), name='w_R_%d_%d' % (i, C+1))})
    biasesR.update({'b_R_%d_%d' % (i, C+1): tf.Variable(tf.zeros([G], name='b_R_%d_%d' % (i, C+1)))})

weightsS = {
    'w_S_1': tf.Variable(tf.random_normal([ks, ks, c_dim, G0], stddev=np.sqrt(2.0 / ks ** 2 / 3)),
                         name='w_S_1'),
    'w_S_2': tf.Variable(tf.random_normal([ks, ks, G0, G], stddev=np.sqrt(2.0 / ks ** 2 / 64)), name='w_S_2') }
biasesS = {
    'b_S_1': tf.Variable(tf.zeros([G0], name='b_S_1')),
    'b_S_2': tf.Variable(tf.zeros([G], name='b_S_2')) }

weightsU = {
    'w_U_1': tf.Variable(tf.random_normal([5, 5, G0, 64], stddev=np.sqrt(2.0 / 25 / G0)), name='w_U_1'),
    'w_U_2': tf.Variable(tf.random_normal([3, 3, 64, 32], stddev=np.sqrt(2.0 / 9 / 64)), name='w_U_2'),
    'w_U_3': tf.Variable(
        tf.random_normal([3, 3, 32, c_dim * scale * scale], stddev=np.sqrt(2.0 / 9 / 32)),
        name='w_U_3') }
biasesU = {
    'b_U_1': tf.Variable(tf.zeros([64], name='b_U_1')),
    'b_U_2': tf.Variable(tf.zeros([32], name='b_U_2')),
    'b_U_3': tf.Variable(tf.zeros([c_dim * scale * scale], name='b_U_3')) }

weight_final = tf.Variable(tf.random_normal([ks, ks, c_dim, c_dim], stddev=np.sqrt(2.0/9/3)), name='w_f')
bias_final = tf.Variable(tf.zeros([c_dim], name='b_f'))

# NOTE: train with batch size
def _phase_shift(I, r):
	return tf.depth_to_space(I, r)

def PS(X, r, color=False):
	if color:
		Xc = tf.split(X, 3, 3)
		X = tf.concat([_phase_shift(x, r) for x in Xc],3)
	else:
		X = _phase_shift(X, r)
        return X

def UPN(input_layer):
    x = tf.nn.conv2d(input_layer, weightsU['w_U_1'], strides=[1,1,1,1], padding='SAME') + biasesU['b_U_1']
    x = tf.nn.relu(x)
    x = tf.nn.conv2d(x, weightsU['w_U_2'], strides=[1,1,1,1], padding='SAME') + biasesU['b_U_2']
    x = tf.nn.relu(x)
    x = tf.nn.conv2d(x, weightsU['w_U_3'], strides=[1,1,1,1], padding='SAME') + biasesU['b_U_3']

    x = PS(x, scale, True)

    return x

def RDBs(input_layer):
    rdb_concat = list()
    rdb_in = input_layer
    for i in range(1, D+1):
        x = rdb_in
        for j in range(1, C+1):
            tmp = tf.nn.conv2d(x, weightsR['w_R_%d_%d' %(i, j)], strides=[1,1,1,1], padding='SAME') + biasesR['b_R_%d_%d' % (i, j)]
            tmp = tf.nn.relu(tmp)
            x = tf.concat([x, tmp], axis=3)

        x = tf.nn.conv2d(x, weightsR['w_R_%d_%d' % (i, C+1)], strides=[1,1,1,1], padding='SAME') +  biasesR['b_R_%d_%d' % (i, C+1)]
        rdb_in = tf.add(x, rdb_in)
        rdb_concat.append(rdb_in)

    return tf.concat(rdb_concat, axis=3)

def model(images):
    images = images/255.0
    images_mean = tf.reduce_mean(images)
    images = images - images_mean

    F_1 = tf.nn.conv2d(images, weightsS['w_S_1'], strides=[1,1,1,1], padding='SAME') + biasesS['b_S_1']
    F0 = tf.nn.conv2d(F_1, weightsS['w_S_2'], strides=[1,1,1,1], padding='SAME') + biasesS['b_S_2']
    
    FD = RDBs(F0)
    
    FGF1 = tf.nn.conv2d(FD, weightsD['w_D_1'], strides=[1,1,1,1], padding='SAME') + biasesD['b_D_1']
    FGF2 = tf.nn.conv2d(FGF1, weightsD['w_D_2'], strides=[1,1,1,1], padding='SAME') + biasesD['b_D_2']
    
    FDF = tf.add(FGF2, F_1)    

    FU = UPN(FDF)
    
    IHR = tf.nn.conv2d(FU, weight_final, strides=[1,1,1,1], padding='SAME') + bias_final

    IHR = tf.clip_by_value((IHR + images_mean)*255.0, 0.0, 255.0)

    return IHR

