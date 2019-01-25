import csv
import os

import numpy as np
import skimage.io as io
from keras.models import Sequential
from keras.layers import Conv2D, Activation, MaxPooling2D, Flatten, Dense
from keras.optimizers import Adam
from keras.utils import np_utils
# from sklearn.model_selection import train_test_split
from sklearn.cross_validation import train_test_split
import gc
import matplotlib.pyplot as plt


#load labels
#load imgs
def loadData():
    csv_path = '../data/malignancy_labels.csv'
    # csv_path = '../data/test_uid_img_roi_labels.csv' #TEST
    # csv_path = '../data/uid_img_roi_labels.csv'
    Y_train = []
    with open(csv_path, 'r') as f1:
        reader = csv.reader(f1)
        for row in reader:
            Y_train.append(row)
        f1.close()
    str = "../data/Test_img/*.jpg"  #TEST
    # str = "../data/img_data3/*.jpg"
    coll = io.ImageCollection(str)
    print(len(coll))
    X_train = []
    for i in range(0, len(coll)):
        print(i + 1)
        img = coll[i]
        # print(img.shape)
        # jishenbianhao3 = []  # 将单通道的灰度图像，变成三通道的灰度图像，因为这样的检测效果会更好
        # for x in range(3):
        #     jishenbianhao3.append(img)
        # jishenbianhao3 = np.array(jishenbianhao3).transpose([1, 2, 0])
        # img1 = jishenbianhao3
        # print(img1.shape)
        X_train.append(img)
    X = np.array(X_train)
    Y = np.array(Y_train)
    # index = np.arange(400)#TEST
    _X_train, _X_test, _Y_train, _Y_test = train_test_split(X, Y, test_size=0.30, random_state=42)
    del X_train,Y_train
    print(gc.collect())
    return _X_train, _Y_train,_X_test,_Y_test
    """
    index = np.arange(17864)
    np.random.shuffle(index)
    _X_train,_Y_train = [],[]
    _X_test,_Y_test = [],[]
    for i in index:
        _X_train.append(np.reshape(X_train[i], (512, 512, 3)))
        _Y_train.append(Y_train[i])
    # for j in range(400, 512):#TEST
    for j in range(17864,19864):
        _X_test.append(np.reshape(X_train[j], (512, 512, 3)))
        _Y_test.append(Y_train[j])
    return _X_train, _Y_train,_X_test,_Y_test
    """

# def loadData2(coll,Y_train):
#     index = np.arange(len(coll))
#     np.random.shuffle(index)
#     for i in index:
#         image = np.reshape(coll[i], (512, 512, 3))
#         label = Y_train[i]
#         yield image, label
def train_vis(history):
    loss = history.history['loss']
    acc = history.history['acc']
    #figure
    fig = plt.figure(figsize=(8,4))
    ax1 = fig.add_subplot(121)
    ax1.plot(loss,label='train_loss')
    ax1.set_xlabel('epochs')
    ax1.set_ylabel('loss')
    ax1.legend()
    ax2 = fig.add_subplot(122)
    ax2.plot(acc,label='train_acc')
    ax2.set_xlabel('epochs')
    ax2.set_ylabel('acc')
    ax2.legend()
    plt.tight_layout()


X_train, Y_train, X_test, Y_test = loadData()
print(len(X_train),len(X_test))
X_train = X_train.reshape(len(X_train), 512, 512, 1)
X_test = X_test.reshape(len(X_test), 512, 512, 1)
print(len(X_train))
model = Sequential()

# model.add(Convolution2D(
#     nb_filter=4,  # 第一层卷积层中滤波器的个数#
#     nb_row=2,  # 滤波器的长度为5#
#     nb_col=2,  # 滤波器的宽度为5#
#     border_mode='same',  # padding mode 为same#
#     input_shape=(512,512,3),
# ))
# model.add(Activation('relu'))  # 激活函数为relu
#
# model.add(MaxPooling2D(
#     pool_size=(2, 2),  # 下采样格为2*2
#     strides=(2, 2),
#     padding='same',  # padding mode is 'same'
# ))
#conv1 relu2 pool1
model.add(Conv2D(4, (13, 13),strides=(2,2), padding='valid',activation = 'relu',input_shape=(512,512,1)))
model.add(MaxPooling2D(strides=(2, 2), padding='valid'))
#conv2 relu2 pool2
model.add(Conv2D(8, (7, 7),strides=(2,2), padding='valid'))
model.add(MaxPooling2D(strides=(2, 2), padding='valid'))
#conv3 relu3
model.add(Conv2D(16, (3, 3),strides=(2,2), padding='valid',activation = 'relu'))
#conv4 relu4
model.add(Conv2D(32, (3, 3),strides=(2,2), padding='valid',activation = 'relu'))
#relu5 pool5
model.add(Activation('relu'))
model.add(MaxPooling2D(strides=(2, 2), padding='valid'))
#fc6 relu6
model.add(Flatten())  # 将多维的输入一维化
model.add(Dense(1052))
model.add(Activation('relu'))
#fc7 relu7
model.add(Dense(500))
model.add(Activation('relu'))
#fc8
model.add(Dense(6))

model.add(Activation('softmax'))  # softmax 用于分类

adam = Adam()  # 学习速率lr=0.0001

model.compile(optimizer=adam,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

print('Training')
history = model.fit(X_train, Y_train, epochs=100, batch_size=64)  # 全部训练次数1次，每次训练批次大小64

outputs_txt = '../outputs/ct_feature0_1.txt'
if os.path.exists(outputs_txt):
    os.unlink(outputs_txt)

with open(outputs_txt,'w')as f1:
    f1.write(str(history.history))

print('Testing')
loss, accuracy = model.evaluate(X_test, Y_test)
print('\nTest loss:', loss)
print('\nTest accuracy', accuracy)
# train_vis(history)
#------------------
plt.plot(history.history['loss'])
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','test'],loc='upper left')
plt.show()

