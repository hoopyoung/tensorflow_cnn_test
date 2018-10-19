#coding:utf-8
# 使用CNN模型训练
import os
import numpy as np
from PIL import Image
import tensorflow as tf
import random
import time

CAPTCHA_LENGTH = 4
CAPTCHA_IMAGE_WIDHT = 176
CAPTCHA_IMAGE_HEIGHT = 56

#验证码字符集，数量
CHARSET_LEN = 36

TRAIN_IMG_PATH = os.path.join(os.path.abspath(os.path.dirname(__file__)),'pic','train')
MODEL_SAVE_PATH = os.path.join(os.path.abspath(os.path.dirname(__file__)),'module')
TRAINING_IMAGE_NAME = []
VALIDATION_IMAGE_NAME = []


def name2lable(name):
    """
    #  将验证码的值转成长度为36*4的数组
    #  假设验证码的值为 abcd
    #  表示a  0 0 0 0 0 0 0 0 0 0 1 0 0 ...
    #  即数组前36位，第11位为1，其他35位为0
    #  以ord(0)的值为基准，即 ord=48, 1 0 0 0 0 0 ... 表示0
    :param name: 验证码的值
    :return: 二维数组
    """
    lable = np.zeros(CHARSET_LEN * CAPTCHA_LENGTH)
    for i,v in enumerate(name):
        _v = ord(v) - ord('0')
        if _v > 9:
            lable[i*CHARSET_LEN + 10 + _v % (ord('a') - ord('0'))] = 1
        else:
            lable[i*CHARSET_LEN +_v] = 1
    return lable

def get_image_file_name(imgPath=TRAIN_IMG_PATH):
    fileName = []
    total = 0
    for filePath in os.listdir(imgPath):
        captcha_name = filePath.split(os.sep)[-1]
        fileName.append(captcha_name)
        total += 1
    return fileName, total

def get_img_data_lable(filename):
    """
    返回图片的数组表示
    其中：
        img_data 为灰度图的数组表示
        img_lable 为验证码值的数组表示
    :return:
    """
    captcha_code = filename.split('.')[0]
    _file = TRAIN_IMG_PATH + os.sep + filename
    img = Image.open(_file)
    img = img.convert('L')
    img_array = np.array(img)
    img_data = img_array.flatten()/255.0
    img_label = name2lable(captcha_code)
    return img_data,img_label

def get_next_batch(batchSize=32,trainOrTest='train',step=0):
    batch_data = np.zeros([batchSize,CAPTCHA_IMAGE_WIDHT*CAPTCHA_IMAGE_HEIGHT])
    batch_lable = np.zeros([batchSize,CHARSET_LEN * CAPTCHA_LENGTH])
    fileNameList = TRAINING_IMAGE_NAME
    if trainOrTest == 'validate':
        fileNameList = VALIDATION_IMAGE_NAME

    totalNumber = len(fileNameList)
    indexStart = step * batchSize
    for i in range(batchSize):
        index = (i + indexStart) % totalNumber
        name = fileNameList[index]
        img_data,img_lable = get_img_data_lable(name)
        batch_data[i,:] = img_data
        batch_lable[i,:] = img_lable
    return batch_data,batch_lable


def train_data_with_CNN():
    # 初始化权重
    def weight_variable(shape, name='weight'):
        init = tf.truncated_normal(shape, stddev=0.1)
        var = tf.Variable(initial_value=init, name=name)
        return var

    # 初始化偏移量
    def bias_variable(shape, name='bias'):
        init = tf.constant(0.1, shape=shape)
        var = tf.Variable(init, name=name)
        return var

    # 卷积
    def conv2d(x, W, name='conv2d'):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME', name=name)

    # 池化
    def max_pool_2X2(x, name='maxpool'):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    # 输入层
    # 请注意 X 的 name，在测试model时会用到它
    X = tf.placeholder(tf.float32, [None, CAPTCHA_IMAGE_WIDHT * CAPTCHA_IMAGE_HEIGHT], name='data-input')
    Y = tf.placeholder(tf.float32, [None, CAPTCHA_LENGTH * CHARSET_LEN], name='label-input')
    x_input = tf.reshape(X, [-1, CAPTCHA_IMAGE_HEIGHT, CAPTCHA_IMAGE_WIDHT, 1], name='x-input')
    #dropout,防止过拟合
    #请注意 keep_prob 的 name，在测试model时会用到它
    keep_prob = tf.placeholder(tf.float32,name='keep-prob')

    # 第一层卷积
    W_conv1 = weight_variable([5, 5, 1, 32], 'W_conv1')
    B_conv1 = bias_variable([32], 'B_conv1')
    conv1 = tf.nn.relu(conv2d(x_input, W_conv1, 'conv1') + B_conv1)
    conv1 = max_pool_2X2(conv1, 'conv1-pool')
    conv1 = tf.nn.dropout(conv1, keep_prob)

    #第二层卷积
    W_conv2 = weight_variable([5,5,32,64], 'W_conv2')
    B_conv2 = bias_variable([64], 'B_conv2')
    conv2 = tf.nn.relu(conv2d(conv1, W_conv2,'conv2') + B_conv2)
    conv2 = max_pool_2X2(conv2, 'conv2-pool')
    conv2 = tf.nn.dropout(conv2, keep_prob)

    #第三层卷积
    W_conv3 = weight_variable([5,5,64,64], 'W_conv3')
    B_conv3 = bias_variable([64], 'B_conv3')
    conv3 = tf.nn.relu(conv2d(conv2, W_conv3, 'conv3') + B_conv3)
    conv3 = max_pool_2X2(conv3, 'conv3-pool')
    conv3 = tf.nn.dropout(conv3, keep_prob)

    # 全链接层
    # 每次池化后，图片的宽度和高度均缩小为原来的一半，进过上面的三次池化，宽度和高度均缩小8倍
    W_fc1 = weight_variable([int(CAPTCHA_IMAGE_WIDHT/8) * int(CAPTCHA_IMAGE_HEIGHT/8) * 64, 1024], 'W_fc1')
    B_fc1 = bias_variable([1024], 'B_fc1')
    fc1 = tf.reshape(conv3, [-1, int(CAPTCHA_IMAGE_WIDHT/8) * int(CAPTCHA_IMAGE_HEIGHT/8) * 64])
    fc1 = tf.nn.relu(tf.add(tf.matmul(fc1, W_fc1), B_fc1))
    fc1 = tf.nn.dropout(fc1, keep_prob)

    # 输出层
    W_fc2 = weight_variable([1024, CAPTCHA_LENGTH * CHARSET_LEN], 'W_fc2')
    B_fc2 = bias_variable([CAPTCHA_LENGTH * CHARSET_LEN], 'B_fc2')
    output = tf.add(tf.matmul(fc1, W_fc2), B_fc2, 'output')

    cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=Y, logits=output))
    train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

    predict = tf.reshape(output, [-1, CAPTCHA_LENGTH, CHARSET_LEN], name='predict')
    labels = tf.reshape(Y, [-1, CAPTCHA_LENGTH, CHARSET_LEN], name='labels')


    # 预测结果
    # 请注意 predict_max_idx 的 name，在测试model时会用到它
    predict_max_idx = tf.argmax(predict, axis=2, name='predict_max_idx')
    labels_max_idx = tf.argmax(labels, axis=2, name='labels_max_idx')
    predict_correct_vec = tf.equal(predict_max_idx, labels_max_idx)
    accuracy = tf.reduce_mean(tf.cast(predict_correct_vec, tf.float32))

    saver = tf.train.Saver()
    gpu_options = tf.GPUOptions(allow_growth = True)
    with  tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        sess.run(tf.global_variables_initializer())
        steps = 0
        for epoch in range(300000):
            train_data, train_label = get_next_batch(64, 'train', steps)
            _,loss = sess.run([train_step,cross_entropy], feed_dict={X: train_data, Y: train_label, keep_prob: 0.75})

            if steps % 100 == 0  :
                print('loss=%f' % loss, end=',')
                test_data, test_label = get_next_batch(128, 'validate', steps)
                acc = sess.run(accuracy, feed_dict={X: test_data, Y: test_label, keep_prob: 1.0})
                print("steps=%d, accuracy=%f" % (steps, acc))
                if acc > 0.99:
                    saver.save(sess, MODEL_SAVE_PATH + os.sep +"crack_captcha.model", global_step=steps)
                    break
            steps += 1

if __name__=='__main__':

    image_filename_list, total = get_image_file_name()
    random.seed(time.time())
    #打乱顺序
    random.shuffle(image_filename_list)
    trainImageNumber = int(total * 0.8)
    #分成测试集
    TRAINING_IMAGE_NAME = image_filename_list[ : trainImageNumber]
    #和验证集
    VALIDATION_IMAGE_NAME = image_filename_list[trainImageNumber : ]

    train_data_with_CNN()
    print('Training finished')





