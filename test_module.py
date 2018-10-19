#coding:utf-8
# 测试训练好的模型
import os
import tensorflow as tf
from PIL import Image
import numpy as np
import requests
import uuid
import random
import time

MODEL_SAVE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),'module')
MODEL_META = 'crack_captcha.model-5300.meta'
TEST_IMG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),'pic','test')
TRAIN_IMG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),'pic','train')
CHARSET_LEN = 36
CAPTCHA_LENGTH = 4

def label2name(lable):
    """
    将一维数组转成验证码值
    0-9表示0-9
    10表示a
    11表示b
    :param lable: 一个验证码值的一维数组 [23 17 12 23]
    :return:
    """
    if len(lable)!= CAPTCHA_LENGTH:
        return None
    name = []
    for i,v in enumerate(lable):
        if v<=9:
            name.append(str(v))
        else:
            v = chr(ord('a') + v - 10)
            name.append(v)
    return name

def get_img_data(filename):
    """
    返回图片的数组表示
    其中：
        img_data 为灰度图的数组表示
    :return:
    """
    img = Image.open(filename)
    img = img.convert('L')
    img_array = np.array(img)
    img_data = img_array.flatten()/255
    return img_data

def download_img(num=1):
    """
    从网站上下载验证码图片
    :param num:
    :return:
    """
    print('正在下载图片保存到pic/test:')
    for i in range(num):
        r = requests.get('http://jwxt.imu.edu.cn/img/captcha.jpg',timeout=10)
        with open(TEST_IMG_PATH + os.sep + '{0}.jpg'.format(uuid.uuid4()),'wb') as f:
            f.write(r.content)
        print('\r 下载进度：{0}/{1}'.format(i+1,num),end='')

def resize_img(_file):
    img = Image.open(_file)
    w,h = img.size
    new_img = img.crop([2,2,w-2,h-2])
    new_img.save(_file)

def resize_all_imgs():
    for _file in os.listdir(TEST_IMG_PATH):
        _file = TEST_IMG_PATH + os.sep + _file
        resize_img(_file)

def test_train_img():
    """
    拿标记好的样本来测试
    :return:
    """
    # 加载graph
    saver = tf.train.import_meta_graph(MODEL_SAVE_PATH + os.sep + MODEL_META)
    graph = tf.get_default_graph()
    # 从graph取得 tensor，他们的name是在构建graph时定义的(查看上面第2步里的代码)
    input_holder = graph.get_tensor_by_name("data-input:0")
    keep_prob_holder = graph.get_tensor_by_name("keep-prob:0")
    predict_max_idx = graph.get_tensor_by_name("predict_max_idx:0")

    train_img_lists = os.listdir(TRAIN_IMG_PATH)
    random.seed(time.time())
    #打乱顺序
    random.shuffle(train_img_lists)
    train_sample = train_img_lists[:10000]

    with tf.Session() as sess:
        saver.restore(sess, tf.train.latest_checkpoint(MODEL_SAVE_PATH))
        count = 0
        total_number = 0
        for filename in train_sample:
            total_number += 1
            right_name = filename.split('.')[0]
            print(filename,end=',')
            filename =  TRAIN_IMG_PATH + os.sep + filename
            img_data = get_img_data(filename)
            predict = sess.run(predict_max_idx, feed_dict={input_holder: [img_data], keep_prob_holder: 1.0})
            predict_name = ''.join(label2name(np.squeeze(predict)))
            if predict_name == right_name:
                result = '正确'
                count += 1
            else:
                result = '错误'
            print('实际值：{}， 预测值：{}，{},测试结果：{}'.format(right_name, predict_name,np.squeeze(predict), result))
            print('\n')
        print('正确率：%.2f%%(%d/%d)' % (count * 100 / total_number, count, total_number))

test_train_img()

def test_download_img():
    """
    从目标站点下载图片测试
    :return:
    """
    download_img(10)
    resize_all_imgs()
    saver = tf.train.import_meta_graph(MODEL_SAVE_PATH + os.sep + MODEL_META)
    graph = tf.get_default_graph()
    # 从graph取得 tensor，他们的name是在构建graph时定义的(查看上面第2步里的代码)
    input_holder = graph.get_tensor_by_name("data-input:0")
    keep_prob_holder = graph.get_tensor_by_name("keep-prob:0")
    predict_max_idx = graph.get_tensor_by_name("predict_max_idx:0")
    gpu_options = tf.GPUOptions(allow_growth = True)
    with  tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        saver.restore(sess,tf.train.latest_checkpoint(MODEL_SAVE_PATH))
        for _file in os.listdir(TEST_IMG_PATH):
            _file = TEST_IMG_PATH + os.sep + _file
            img_data = get_img_data(_file)
            predict = sess.run(predict_max_idx, feed_dict={input_holder: [img_data], keep_prob_holder: 1.0})
            predict_name =  ''.join(label2name(np.squeeze(predict)))
            print(_file,predict_name)
            os.rename(_file,TEST_IMG_PATH + os.sep + predict_name + '.jpg')

