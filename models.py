# test models
# import tensorflow as tf
# import numpy as np
#
# from tensorflow.examples.tutorials.mnist import input_data
#
# epoches = 100
# batchsize = 100
# index = np.array([13] * batchsize).astype(np.int32)
# mnist = input_data.read_data_sets('../data/mnist/', one_hot=True)
#
# IDX = tf.placeholder(tf.int32, shape=[batchsize], name='index')
# X = tf.placeholder(tf.float32, shape=[batchsize, 784], name='input')
# Y = tf.placeholder(tf.float32, shape=[batchsize, 10], name='target')
#
# def conv2d(inputs):
#     return tf.layers.conv2d(inputs, filters=3, kernel_size=(2, 28), strides=[1, 1], data_format='channels_last')
#
#
# def DmaxPooling2d(inputs, idx):
#     s = np.array([]).astype(np.float32)
#     for i in range(inputs.shape[0]):
#         for j in range(inputs.shape[-1]):
#             s = tf.concat([s, [tf.reduce_max(inputs[i, :idx[i], 0, j])],
#                            [tf.reduce_max(inputs[i, idx[i]:inputs.shape[1], 0, j])]], axis=0)
#     return tf.reshape(s, [batchsize, -1])
#
#
# def dense(inputs):
#     s = tf.layers.dense(inputs=inputs, units=10, activation=tf.nn.relu)
#     return s
#
# x_image = tf.reshape(X, shape=[-1, 28, 28, 1])
# conv1 = conv2d(x_image)
#
# pool = DmaxPooling2d(conv1, IDX)
# # pool = tf.layers.max_pooling2d(conv1, [2,1], 1)
# # print(pool.get_shape())
# # pool = tf.reshape(pool, [batchsize, 26*3])
#
# # print(pool.get_shape())
# logits = dense(pool)
#
# entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=Y, name='entropy')
# loss = tf.reduce_mean(entropy, name='loss')
#
# optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss)
#
# preds = tf.nn.softmax(logits)
# correct_preds = tf.equal(tf.argmax(preds, 1), tf.argmax(Y, 1))
# accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32))
#
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     for i in range(epoches):
#         x_input, y_output = mnist.train.next_batch(batchsize)
#         _, loss_, acc = sess.run([optimizer, loss, accuracy], feed_dict={X: x_input, Y: y_output, IDX: index})
#
#         print('epoch{0}:\tloss: {1}\taccuracy:{2}'.format(i, loss_, acc))
#
#     # print(type(conv1))
#     # output = sess.run(conv1, feed_dict={X: x_input, IDX: index})
#     # print(output.shape)
#     # print(type(output))
#     #
#     # output2 = sess.run(pool, feed_dict={X:x_input, IDX:index})
#     # print(output2.shape)
#     # print(type(output2))


import pickle
import numpy as np
import tensorflow as tf


def load_data(file):
    with open(file, 'rb') as f:
        return pickle.load(f)


def conv2d(inputs, window_size=3, k=64):
    return tf.layers.conv2d(inputs=inputs, filters=3, kernel_size=[window_size, k])

def DmaxPooling2d(inputs, idx):
    s = np.array([]).astype(np.float32)
    for i in range(inputs.shape[0]):
        for j in range(inputs.shape[-1]):
            s = tf.concat([s, [tf.reduce_max(inputs[i, :idx[i], 0, j])],
                           [tf.reduce_max(inputs[i, idx[i]:inputs.shape[1], 0, j])]], axis=0)
    # todo
    return tf.reshape(s, [inputs.shape[0], -1])


def dense(inputs, units=34):
    return tf.layers.dense(inputs=inputs, units=units, activation=tf.nn.relu)


def get_batch(data, batch_size):
    c = 0
    for datum in data:
        instLen = datum['instLen']
        embedding = datum['embeddings']
        triggerInfo = datum['triggerInfo']
        # yield instLen, embeddings, triggerInfo

        sentEmbeddings = []
        target = []
        index = []
        for i in range(instLen):
            pad = np.arange(-i, -i + embedding.shape[0])
            embedding = np.column_stack([embedding, pad])
            for trigger in triggerInfo:
                if i == trigger[0]:
                    sentEmbeddings.append(embedding)
                    target.append(trigger[1])
                    index.append(i)
                    c += 1
                    break
            sentEmbeddings.append(embedding)  # 'None' is 0
            target.append(0)
            index.append(i)
            c += 1

            if c == batch_size:
                c = 0
                yield sentEmbeddings, target, index
                sentEmbeddings = []
                target = []
                index = []
        # yield sentEmbeddings, target, index


if __name__ == '__main__':
    # tf.one_hot()

    batch_size = 100
    n_classes = 34
    n_epoches = 5
    maxlen = 117
    k = 64
    dimen = k + 1
    file = '../processed_data/train_data.pkl'
    data = load_data(file)

    generator = get_batch(data=data, batch_size=batch_size)

    # todo
    x, y, idx = next(generator)
    print(len(x))
    print(len(y))
    print(len(idx))

    x, y, idx = next(generator)
    print(len(x))
    print(len(y))
    print(len(idx))

    x, y, idx = next(generator)
    print(len(x))
    print(len(y))
    print(y)
    print(len(idx))

    exit()

    X = tf.placeholder(tf.float32, shape=[batch_size, maxlen, dimen], name='X')
    Y = tf.placeholder(tf.int32, shape=None, name='y')
    IDX = tf.placeholder(tf.int32, name='idx')

    x_reshape = tf.reshape(X, shape=[batch_size, maxlen, dimen, 1])
    y_onehot = tf.one_hot(Y, n_classes)

    conv1 = conv2d(x_reshape, k=dimen)

    # todo
    # if IDX == 0:
    #     IDX = 1
    # if IDX == maxlen:
    #     IDX -= 2

    pool = DmaxPooling2d(conv1, idx=IDX)
    logits = dense(pool, n_classes)

    entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=y_onehot)
    loss = tf.reduce_mean(entropy)

    optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for i in range(n_epoches):
            x, y, idx = next(generator)
            _, loss_ = sess.run([optimizer, loss], feed_dict={X: x, Y: y, IDX: idx})
