import matplotlib
import glob
import os
import tensorflow as tf
import numpy as np
import time
import matplotlib.image as mpimg
import pandas as pd


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# 读取图片
def read_img(path):
    cate = [path + x for x in os.listdir(path) if os.path.isdir(path + x)]
    imgs = []
    labels = []
    for idx, folder in enumerate(cate):
        for im in glob.glob(folder + '/*.csv'):
            # print('reading the images:%s' % (im))
            # img = mpimg.imread(im)
            data = np.loadtxt(im,delimiter=',')
            img = np.reshape(data,newshape=(64,3,1))
            imgs.append(img)
            labels.append(idx)
    return np.asarray(imgs, np.float), np.asarray(labels, np.int)


def read_img2(path):
    imgs = []
    # for im in glob.glob(path + '/*.png'):
    #     print('reading the images:%s' % (im))
    #     img = mpimg.imread(im)
    #
    #     # img = transform.resize(img, (w, h))
    #     imgs.append(img)

    files = os.listdir(path)  # 采用listdir来读取所有文件
    files.sort(key=lambda x:int(x[:-4]))
    for file_ in files:
        if not os.path.isdir(path + file_):
            img = mpimg.imread(path+file_)
            # print('reading the images:%s' % (path+file_))
            imgs.append(img)

    return np.asarray(imgs, np.float)

def eval_seq(path,sess,save_name):
    num = len([name for name in os.listdir(path) if os.path.isfile(os.path.join(path, name))])
    s0 = np.zeros((num, 2))
    data = read_img2(path)
    data = np.reshape(data, [num, 64, 3, 1])
    pred = sess.run(logits, feed_dict={x: data, y0: s0})
    np.savetxt(save_name, pred, delimiter=",")

# 定义一个函数，按批次取数据
def minibatches(inputs=None, targets=None, batch_size=None, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batch_size + 1, batch_size):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batch_size]
        else:
            excerpt = slice(start_idx, start_idx + batch_size)
        yield inputs[excerpt], targets[excerpt]

def rnn_basic(_x,_w1,_b1):
    h1 = tf.nn.sigmoid(tf.add(tf.matmul(_x, _w1), _b1))
    return h1

def rnn_output(_s1,_x,_s2,_w,_w2,_b1,_b2):
    _sxs = tf.concat([_s1,_x,_s2],1)
    h1 = rnn_basic(_sxs,_w,_b1)
    lgts=tf.add(tf.matmul(h1, _w2), _b2)
    output = tf.nn.softmax(lgts)
    return output,lgts

def rnn_state(_e1,_e2, _w, _b1):
    _e12 = tf.concat([_e1,_e2],1)
    state = rnn_basic(_e12,_w,_b1)
    return state

def networks_model2(s,_x1,_x2,_x3, _weight,_biase):

    w_pc = tf.concat([_weight['pre'],_weight['cur']],0)
    w_cn = tf.concat([_weight['cur'],_weight['next']],0)
    w_pcn = tf.concat([w_pc,_weight['next']],0)

    _s2 = rnn_state(s,_x1,w_pc,_biase['b1'])
    _s3 = rnn_state(_s2,_x2,w_pc,_biase['b1'])

    s2_ = rnn_state(_x3,s,w_cn,_biase['b1'])
    s1_ = rnn_state(_x2,s2_,w_cn,_biase['b1'])

    _y1,_ = rnn_output(s,_x1,s1_,w_pcn,_weight['out'],_biase['b1'],_biase['out'])
    _y2, lgts = rnn_output(_s2,_x2,s2_,w_pcn,_weight['out'],_biase['b1'],_biase['out'])
    _y3, _ = rnn_output(_s3,_x3,s,w_pcn,_weight['out'],_biase['b1'],_biase['out'])

    return _y1,_y2,_y3,lgts


# --------------------------- 生成训练测试数据 -----------------------------------
path = 'd:\\Program Files/MATLAB/NeuroTIS-master/Coding-VBRNN/Train_coding20000_84human/'
test_path = 'd:\\Program Files/MATLAB/NeuroTIS-master/Coding-VBRNN/Test_coding4842_84human/'


# 将所有的图片resize成100*100
w = 3
h = 64
c = 1
pw = 1.0
n_epoch = 60
batch_size = 500


train_dir0 = path + '/0/'
train_dir1 = path + '/1/'

test_dir0 = test_path + '/0/'
test_dir1 = test_path + '/1/'


train_l0 = len([name for name in os.listdir(train_dir0) if os.path.isfile(os.path.join(train_dir0, name))])
train_l1 = len([name for name in os.listdir(train_dir1) if os.path.isfile(os.path.join(train_dir1, name))])
test_l0 = len([name for name in os.listdir(test_dir0) if os.path.isfile(os.path.join(test_dir0, name))])
test_l1 = len([name for name in os.listdir(test_dir1) if os.path.isfile(os.path.join(test_dir1, name))])

data, label = read_img(path)
data = np.reshape(data,[train_l0+train_l1,h,w,1])
test_data, test_label = read_img(test_path)
test_data = np.reshape(test_data, [test_l0+test_l1, h, w, 1])



# ------------------------------- 构建网络 -------------------------------------
# 占位符
x = tf.placeholder(tf.float32, shape=[None, h, w, c], name='x')
# x2 = tf.placeholder(tf.float32,shape=[None, h2],name = 'x2')
y_ = tf.placeholder(tf.int32, shape=[None, ], name='y_')
y0 = tf.placeholder(tf.float32, [None, 5], name= 'y0')

x1, x2, x3 = tf.split(x, [1, 1, 1 ], 2)
xs1 = tf.reshape(x1, [-1, 64])
xs2 = tf.reshape(x2, [-1, 64])
xs3 = tf.reshape(x3, [-1, 64])

weights = {

    'pre': tf.Variable(tf.truncated_normal([5, 5], stddev=0.01)),
    'cur': tf.Variable(tf.truncated_normal([64, 5], stddev=0.01)),
    'next': tf.Variable(tf.truncated_normal([5, 5], stddev=0.01)),
    'out': tf.Variable(tf.truncated_normal([5, 2], stddev=0.01))
}

biases = {
    'b1': tf.Variable(tf.zeros([5])),
    'out': tf.Variable(tf.zeros([2]))
}

y1,y2,y3,logits = networks_model2(y0,xs1,xs2,xs3,weights,biases)

loss = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(targets=tf.one_hot(y_,2),logits=logits,pos_weight=pw))

train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
correct_prediction = tf.equal(tf.cast(tf.argmax(logits, 1), tf.int32), y_)
acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# auc = tf.metrics.auc(y_[:,1],logits[:,1])

# 训练和测试数据，可将n_epoch设置更大一些
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
for epoch in range(n_epoch):
    start_time = time.time()
    state0 = np.zeros((batch_size, 5))
    # training
    train_loss, train_acc, train_batch = 0, 0, 0
    for x_train_a, y_train_a in minibatches(data, label, batch_size, shuffle=True):
        _, err, ac = sess.run([train_op, loss, acc], feed_dict={x: x_train_a, y_: y_train_a ,y0: state0})
        train_loss += err
        train_acc += ac
        train_batch += 1

    # validation
    val_loss, val_acc, val_batch = 0, 0, 0
    for x_val_a, y_val_a in minibatches(test_data, test_label, batch_size, shuffle=False):
        err, ac= sess.run([loss, acc], feed_dict={x: x_val_a, y_: y_val_a ,y0: state0})
        val_loss += err
        val_acc += ac
        val_batch += 1
    print("(%d/%d) train loss: %f, train acc: %f, validation loss: %f ,validation acc: %f"
          %(n_epoch, epoch+1 ,train_loss / train_batch,train_acc / train_batch,val_loss / val_batch, val_acc / val_batch))
s0 = np.zeros((test_l0 + test_l1, 5))
pred = sess.run(logits, feed_dict={x: test_data, y_: test_label, y0: s0})
print(pred)

# for i in range(1,1284,1):
#     seq_dir = 'd:\\Program Files/MATLAB/NeuroTIS/G2000/whole1283_G2000_84/' + str(i) + '/'
#     save_name = 'codingcsv1283_84/' + str(i) + '.csv'
#     eval_seq(seq_dir, sess, save_name)
#
# for i in range(1,179,1):
#     seq_dir = 'd:\\Program Files/MATLAB/NeuroTIS/G2000/whole178_G2000_84/' + str(i) + '/'
#     save_name = 'codingcsv178_84/' + str(i) + '.csv'
#     eval_seq(seq_dir, sess, save_name)

np.savetxt("pred93.csv", pred, delimiter=",")
np.savetxt("test_label93.csv", test_label, delimiter=",")
sess.close()
