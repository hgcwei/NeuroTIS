import matplotlib
import glob
import os
import tensorflow as tf
import numpy as np
import time
import matplotlib.image as mpimg
# import scipy.io

with tf.device('/cpu:0'):

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    path = 'D:\\tis/train1_22atg6_1/'
    test_path = 'D:\\tis/test1_22atg6_1/'

    # 将所有的图片resize成100*100
    w = 6
    h = 401
    c = 1
    batch_size = 200
    # n_epoch = 13
    # pw = 3.0

    n_epoch = 20
    pw = 1.0


    train_dir0 = path + '/0/'
    train_dir1 = path + '/1/'

    test_dir0 = test_path + '/0/'
    test_dir1 = test_path + '/1/'

    train_l0 = len([name for name in os.listdir(train_dir0) if os.path.isfile(os.path.join(train_dir0, name))])
    train_l1 = len([name for name in os.listdir(train_dir1) if os.path.isfile(os.path.join(train_dir1, name))])
    test_l0 = len([name for name in os.listdir(test_dir0) if os.path.isfile(os.path.join(test_dir0, name))])
    test_l1 = len([name for name in os.listdir(test_dir1) if os.path.isfile(os.path.join(test_dir1, name))])
    # train_num1 = len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))])


    # 读取图片
    def read_img(path):
        cate = [path + x for x in os.listdir(path) if os.path.isdir(path + x)]
        imgs = []
        labels = []
        for idx, folder in enumerate(cate):
            for im in glob.glob(folder + '/*.csv'):
                # print('reading the images:%s' % (im))
                # img = mpimg.imread(im)
                mat = np.loadtxt(open(im, "rb"), delimiter=",", skiprows=0)
                # img = transform.resize(img, (w, h))
                imgs.append(mat)
                labels.append(idx)
        return np.asarray(imgs, np.float32), np.asarray(labels, np.int32)



    data, label = read_img(path)
    # 打乱顺序
    # num_example = data.shape[0]
    # arr = np.arange(num_example)
    # np.random.shuffle(arr)
    # data = data[arr]
    # data = np.reshape(data,[30654,h,w,1])
    data = np.reshape(data, [train_l0+train_l1, h, w, 1])
    # data = np.reshape(data, [14868, h, w, 1])
    # label = label[arr]
    # 将所有数据分为训练集和验证集
    # ratio = 0.8
    # s = np.int(num_example * ratio)
    # x_train = data[:s]
    # y_train = label[:s]
    # x_val = data[s:]
    # y_val = label[s:]

    test_data, test_label = read_img(test_path)
    print(test_label)

    test_data = np.reshape(test_data, [test_l0+test_l1, h, w, 1])

    # data2 = np.transpose(scipy.io.loadmat('data2.mat'))
    # test_data2 = np.transpose(scipy.io.loadmat('test_data.mat'))

    # data2 = np.loadtxt(open("trainxSS.csv","rb"),delimiter=",",skiprows=0)
    # test_data2 = np.loadtxt(open("testxSS.csv","rb"),delimiter=",",skiprows=0)




    #
    # label = np.int32(np.hstack((np.transpose(np.matrix(label)),1-np.transpose(np.matrix(label)))) )
    # test_label = np.int32(np.hstack((np.transpose(np.matrix(test_label)), 1 - np.transpose(np.matrix(test_label)))))
    #





    print(test_label)
    # print(data.shape,data2.shape,test_data.shape,test_data2.shape)

    # -----------------构建网络----------------------
    # 占位符
    x = tf.placeholder(tf.float32, shape=[None, h, w, c], name='x')
    # x2 = tf.placeholder(tf.float32,shape=[None, h2],name = 'x2')
    y_ = tf.placeholder(tf.int32, shape=[None, ], name='y_')
    x1, x2 = tf.split(x, [2, 4], 2)
    x21, x22, x23= tf.split(x2,[50,301,50],1)
    # 第一个卷积层（401——>198)
    conv1 = tf.layers.conv2d(
        inputs=x22,
        filters=100,
        kernel_size=[8, 4],
        padding="valid",
        activation=tf.nn.relu,
        kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[3, 1], strides=3)
    drop1 = tf.layers.dropout(pool1,0.3)
    # 第二个卷积层(198->96)
    conv2 = tf.layers.conv2d(
        inputs=drop1,
        filters=100,
        kernel_size=[3, 1],
        padding="valid",
        activation=tf.nn.relu,
        kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[3, 1], strides=3)
    drop2 = tf.layers.dropout(pool2, 0.3)
    # # 第三个卷积层(99->49)
    conv3 = tf.layers.conv2d(
        inputs=drop2,
        filters=100,
        kernel_size=[3, 1],
        padding="valid",
        activation=tf.nn.relu,
        kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
    pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[3, 1], strides=3)
    drop3 = tf.layers.dropout(pool3, 0.3)

    re1 = tf.reshape(drop3, [-1, 10*100])
    # 全连接层
    dense1 = tf.layers.dense(inputs=re1,
                             units=100,
                             activation=tf.nn.relu,
                             kernel_initializer=tf.truncated_normal_initializer(stddev=0.01)
                             )
    drop4 = tf.layers.dropout(dense1, 0.2)
    logits = tf.layers.dense(inputs=drop4,
                             units=2,
                             activation=None,
                             kernel_initializer=tf.truncated_normal_initializer(stddev=0.01)
                             )
    # ---------------------------网络结束---------------------------
    # tf.nn.weighted_cross_entropy_with_logits
    # loss = tf.nn.weighted_cross_entropy_with_logits(targets=y_,logits=logits,pos_weight=0.7)
    # loss = tf.losses.sparse_softmax_cross_entropy(labels=y_, logits=logits, weights=0.1)

    loss = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(targets=tf.one_hot(y_,2),logits=logits,pos_weight=pw))

    train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
    correct_prediction = tf.equal(tf.cast(tf.argmax(logits, 1), tf.int32), y_)
    acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    # auc = tf.metrics.auc(y_[:,1],logits[:,1])

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


    # 训练和测试数据，可将n_epoch设置更大一些


    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    for epoch in range(n_epoch):
        start_time = time.time()

        # training
        train_loss, train_acc, train_batch = 0, 0, 0
        for x_train_a, y_train_a in minibatches(data, label, batch_size, shuffle=True):
            _, err, ac = sess.run([train_op, loss, acc], feed_dict={x: x_train_a, y_: y_train_a})
            train_loss += err
            train_acc += ac
            train_batch += 1
        # validation
        val_loss, val_acc, val_batch = 0, 0, 0
        for x_val_a, y_val_a in minibatches(test_data, test_label, batch_size, shuffle=False):
            err, ac= sess.run([loss, acc], feed_dict={x: x_val_a, y_: y_val_a})
            val_loss += err
            val_acc += ac
            val_batch += 1
        print("(%d/%d) train loss: %f, train acc: %f, validation loss: %f ,validation acc: %f"
              %(n_epoch, epoch+1 ,train_loss / train_batch,train_acc / train_batch,val_loss / val_batch, val_acc / val_batch))

    pred = sess.run(logits, feed_dict={x: test_data, y_: test_label})
    print(pred)
    np.savetxt("pred1_3.csv", pred, delimiter=",")
    np.savetxt("test_label1_3.csv", test_label, delimiter=",")
    sess.close()
