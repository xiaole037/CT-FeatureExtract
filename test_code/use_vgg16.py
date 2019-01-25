'''调用vgg16，训练224x224的原始ct图像'''
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from CT_feature_extract.data_preprocess.load_data import loadImgCsv,resizeImg,readImgTxt,loadEvalImgTxt
from CT_feature_extract._Defined_Net.VGG16 import Vgg16


class Vgg16:
    vgg_mean = [103.939, 116.779, 123.68]

    def __init__(self, vgg16_npy_path=None, restore_from=None):
        # pre-trained parameters
        try:
            self.data_dict = np.load(vgg16_npy_path, encoding='latin1').item()
        except FileNotFoundError:
            print('Cannot found VGG16 parameters')

        self.tfx = tf.placeholder(tf.float32, [None, 224, 224, 3])
        self.tfy = tf.placeholder(tf.float32, [None, 1])

        # Convert RGB to BGR
        red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=self.tfx * 255.0)
        bgr = tf.concat(axis=3, values=[
            blue - self.vgg_mean[0],
            green - self.vgg_mean[1],
            red - self.vgg_mean[2],
        ])

        # pre-trained VGG layers are fixed in fine-tune
        conv1_1 = self.conv_layer(bgr, "conv1_1")
        conv1_2 = self.conv_layer(conv1_1, "conv1_2")
        pool1 = self.max_pool(conv1_2, 'pool1')

        conv2_1 = self.conv_layer(pool1, "conv2_1")
        conv2_2 = self.conv_layer(conv2_1, "conv2_2")
        pool2 = self.max_pool(conv2_2, 'pool2')

        conv3_1 = self.conv_layer(pool2, "conv3_1")
        conv3_2 = self.conv_layer(conv3_1, "conv3_2")
        conv3_3 = self.conv_layer(conv3_2, "conv3_3")
        pool3 = self.max_pool(conv3_3, 'pool3')

        conv4_1 = self.conv_layer(pool3, "conv4_1")
        conv4_2 = self.conv_layer(conv4_1, "conv4_2")
        conv4_3 = self.conv_layer(conv4_2, "conv4_3")
        pool4 = self.max_pool(conv4_3, 'pool4')

        conv5_1 = self.conv_layer(pool4, "conv5_1")
        conv5_2 = self.conv_layer(conv5_1, "conv5_2")
        conv5_3 = self.conv_layer(conv5_2, "conv5_3")
        pool5 = self.max_pool(conv5_3, 'pool5')

        # detach original VGG fc layers and
        # reconstruct your own fc layers serve for your own purpose
        self.flatten = tf.reshape(pool5, [-1, 7*7*512])
        self.fc6 = tf.layers.dense(self.flatten, 256, tf.nn.relu, name='fc6')
        self.out = tf.layers.dense(self.fc6, 1, name='out')

        self.sess = tf.Session()
        if restore_from:
            saver = tf.train.Saver()
            saver.restore(self.sess, restore_from)
        else:   # training graph
            self.loss = tf.losses.mean_squared_error(labels=self.tfy, predictions=self.out)
            self.train_op = tf.train.RMSPropOptimizer(0.001).minimize(self.loss)
            self.sess.run(tf.global_variables_initializer())

    def max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def conv_layer(self, bottom, name):
        with tf.variable_scope(name):   # CNN's filter is constant, NOT Variable that can be trained
            conv = tf.nn.conv2d(bottom, self.data_dict[name][0], [1, 1, 1, 1], padding='SAME')
            lout = tf.nn.relu(tf.nn.bias_add(conv, self.data_dict[name][1]))
            return lout

    def train(self, x, y):
        loss, _ = self.sess.run([self.loss, self.train_op], {self.tfx: x, self.tfy: y})
        return loss

    def predict(self, paths):
        fig, axs = plt.subplots(1, 2)
        for i, path in enumerate(paths):
            x = resizeImg(path)
            length = self.sess.run(self.out, {self.tfx: x})
            axs[i].imshow(x[0])
            axs[i].set_title('Len: %.1f ' % length)
            axs[i].set_xticks(()); axs[i].set_yticks(())
        plt.show()

    def predictAcc(self, paths,value):
        _true = 0
        _total = 0
        for i, path in enumerate(paths):
            x = resizeImg(path)
            pre_value = self.sess.run(self.out, {self.tfx: x})
            # print(round(float(pre_value[0]),1),round(float(pre_value[0]),0),value[i][0])
            if value[i][0] == round(float(pre_value[0]),0):
                _true += 1
            _total += 1
            precision = round(_true / _total,2)
            recall = round(_true / len(value),2)
            print('acc:',precision,' rec:',recall)


    def save(self, path='./save_model/test_use_vgg16'):
        saver = tf.train.Saver()
        saver.save(self.sess, path, write_meta_graph=False)

def saveLossValue(num,loss_value):
    f1 = open('../outputs/vgg_loss.txt','a+')
    f1.write(str(num)+' '+str(loss_value)+'\n')
    f1.close()

def train():
    # load data
    # imgs,y_label = loadImgCsv(img_root_path, labels_file)#用Test_img测试
    imgs, y_label = readImgTxt(LIDC_data)#用LIDC1训练

    x_img = np.concatenate(imgs, axis=0)
    # y_label = np.concatenate(labels, axis=0)

    vgg = Vgg16(vgg16_npy_path='../_Defined_Net/vgg16_model/vgg16_Morvan.npy')
    print('Net built')
    for i in range(5000):
        b_idx = np.random.randint(0, len(x_img), 6)
        b = x_img[b_idx]
        a = y_label[b_idx]
        train_loss = vgg.train(x_img[b_idx], y_label[b_idx])
        print(i, 'train loss: ', train_loss)
        saveLossValue(i,train_loss)

    vgg.save('./save_model/test_use_vgg16')  # save learned fc layers


def eval(predict_img,true_value):
    vgg = Vgg16(vgg16_npy_path='../_Defined_Net/vgg16_model/vgg16_Morvan.npy',
                restore_from='./save_model/test_use_vgg16')
    vgg.predictAcc(predict_img,true_value)


if __name__=='__main__':
    #用Test_img中的数据测试模型是否跑的通
    img_root_path = '../data/test_img/'
    csv_file = '../data/malignancy_labels.csv'
    labels_file = '../data/malignancy_labels.txt'

    #用LIDC1中的数据训练，Test_img中的数据测试
    # predict_img = ['../data/predict_img/3.jpg','../data/predict_img/12.jpg']
    LIDC_data = '../data/LIDC1/'
    train()
    print('train finished')

    # predict_img, true_value = loadEvalImgTxt('../data/LIDC2/')
    # eval(predict_img,true_value)