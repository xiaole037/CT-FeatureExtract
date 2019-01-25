'''
读取roi图像，并进行填充：roi根绝其自身大小填充成224x224
目前还没调试通
'''
import cv2 as cv
import numpy as np
import os
from PIL import Image

# 泛洪填充(二值图像填充)
def fill_binary(root_path,imgages):
    for i in imgages:
        img = Image.open(root_path+i)
        I = np.array(img)
        print(I.shape)
        # I[100:300, 100:300] = 255
        # cv.imshow("1",I)

        image = np.zeros([500, 500, 3], np.uint8)
        print(image.shape)
        image[100:300, 100:300] = 255
        cv.imshow("fill_binary", image)

        mask = np.ones([502, 502], np.uint8)  # mask要保证比原图像高和宽都多2
        mask[101:301, 101:301] = 0
        cv.floodFill(image, mask, (200, 200), (255, 0, 0), cv.FLOODFILL_MASK_ONLY)  # mask不为0的区域不会被填充，mask为0的区域才会被填充
        cv.imshow("filled_binary", image)


def paddingData(img_path):
    print('……')


if __name__=='__main__':
    root_path = "../data/test_img/"
    img_path = os.listdir(root_path)


    fill_binary(root_path,img_path)
    cv.waitKey(0)
    cv.destroyAllWindows()
    print('padding data finished')