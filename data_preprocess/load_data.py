#
import csv
import gc
import io
import os
import cv2
import numpy as np
import skimage.io
import skimage.transform
import matplotlib.pyplot as plt

from sklearn.cross_validation import train_test_split

'''加载img和csv'''
def loadData(csv_path,img_path):
    Y_train = []
    with open(csv_path, 'r') as f1:
        reader = csv.reader(f1)
        for row in reader:
            Y_train.append(row)
        f1.close()
    coll = io.ImageCollection(img_path)
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
    _X_train, _X_test, _Y_train, _Y_test = train_test_split(X, Y, test_size=0.20, random_state=42)
    del X_train,Y_train
    print(gc.collect())
    return _X_train, _Y_train,_X_test,_Y_test


#resize img 512 to 224
def resizeImg(path):
    # img = skimage.io.imread(path)
    # img = img / 255.0
    img2= cv2.imread(path)
    img2 = img2 / 255.0


    # resize to 224, 224
    resized_img = skimage.transform.resize(img2, (224, 224))[None, :, :, :]   # shape [1, 224, 224, 3]
    # print(resized_img.shape)
    # plt.imshow(img)
    # plt.imshow(resized_img)
    return resized_img


#加载原始团ing，并将原始512尺寸resize成224
def loadImgCsv(img_path,labels_file):
    img_list = []
    img_name= os.listdir(img_path)
    for i in img_name:
        _path = img_path + i
        try:
            resized_img = resizeImg(_path)
        except OSError:
            continue
        img_list.append(resized_img)  # [1, height, width, depth] * n
        # if len(img_list) == 150:  # only use 400 imgs to reduce my memory load
        #     break

    # fake length data for img
    print('x:',len(img_list))

    #读取标签
    labels = np.loadtxt(labels_file)
    labels = np.reshape(labels,(-1,1))
    print(labels.shape)
    # #不用open('file','r')读取，原因：该方法读取到的是list，后面转成array困难
    # _labels = []
    # f1 = open(labels_file,'r')
    # for l in range(len(img_list)):
    #     _labels.append(f1.readline())
    # f1.close()
    # labels = np.array(_labels)
    # if len(labels) > len(img_list):
    #     print('y:',len(labels[:len(img_list)]))
    #     return img_list,labels[:len(img_list)]
    # else:
    #     print('y:',len(labels))
    #     return img_list,labels
    return img_list, labels


#加载图像和对应恶性程度的标注
def loadImgTxt(img_path,label_m,label_file):
    img_list = []
    img_name = os.listdir(img_path)
    for i in img_name:
        _path = img_path + i
        try:
            resized_img = resizeImg(_path)
        except OSError:
            continue
        img_list.append(resized_img)  # [1, height, width, depth] * n
        # if len(img_list) == 150:  # only use 400 imgs to reduce my memory load
        #     break
    for l in range(len(img_name)):
        f_label = open(label_file,'a+')
        f_label.write(str(label_m)+'\n')
        f_label.close()
    return img_list

#返回nodule个数
def noduleNumber(LIDC_data,patient_i,label):
    nodule_num = []
    malignancy = []
    with open(LIDC_data + patient_i + label, 'r') as f_label:
        reader = csv.reader(f_label)
        docter = ''
        for row_l in reader:
            if docter == '':
                nodule_num.append(row_l[9])
                malignancy.append(row_l[8])
                docter = row_l[9][-18:-15]
            elif docter == row_l[9][-18:-15]:
                nodule_num.append(row_l[9])
                malignancy.append(row_l[8])
            else:
                break
    return nodule_num,malignancy


def readImgTxt(LIDC_data):
    LIDC_list = os.listdir(LIDC_data)
    print(len(LIDC_list))
    label_file = '../data/LIDC1_label_m.txt'
    if os.path.isfile(label_file):
        os.remove(label_file)
    img_list = []
    for i in LIDC_list:
        print(i)
        nn,mm = noduleNumber(LIDC_data,i,'/label.csv')
        for (n,m) in zip(nn,mm):
            img_list += loadImgTxt(LIDC_data+n[7:],m,label_file)
        # if len(img_list) > 500:
        #     break
    labels = np.loadtxt(label_file)
    labels = np.reshape(labels,(-1,1))
    print(labels.shape)
    return img_list,labels


def loadEvalImgTxt(LIDC2_data):
    print('perpar eval data...')
    labels_file = '../data/LIDC2_label_m.txt'
    if os.path.isfile(labels_file):
        os.remove(labels_file)
    img_list = []
    LIDC2_list= os.listdir(LIDC2_data)
    for i in LIDC2_list:
        print(i)
        nn,mm = noduleNumber(LIDC2_data,i,'/labels.csv')
        for (n, m) in zip(nn, mm):
            imgs_file= os.listdir(LIDC2_data + n[7:])
            for i_f in imgs_file:
                img_list.append(LIDC2_data + n[7:]+i_f)
            for l in range(len(imgs_file)):
                f_label = open(labels_file, 'a+')
                f_label.write(str(m) + '\n')
                f_label.close()
    labels = np.loadtxt(labels_file)
    labels = np.reshape(labels, (-1, 1))
    # print(labels.shape)
    return img_list, labels


if __name__=='__main__':
    #(1)用Test_img数据测试
    # img_root_path = '../data/Test_img/'
    # labels_file = '../data/malignancy_labels.txt'
    # imgs, labels = loadImgCsv(img_root_path, labels_file)

    #（2）用LIDC1数据
    # LIDC_data = '../data/LIDC1/'
    # imgs, labels = readImgTxt(LIDC_data)

    #（3）用Test_img第一次测试
    # csv_file = '../data/malignancy_labels.csv'
    # img_path = "../data/Test_img/*.jpg"
    # X_train, Y_train, X_test, Y_test = loadData(csv_file,img_path)

    #(4)eval,用数据LIDC2
    predict_img, true_value = loadEvalImgTxt('../data/LIDC2/')

    print('load data finished')