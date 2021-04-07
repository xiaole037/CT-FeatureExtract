"""提取xml文件中的关键信息，以及对应的有效的坐标值
有效坐标值是指坐标数量大于一对的标注，这类标注是结节长径小于3mm的结节的中心点。
关键信息和坐标分开保存到csv文件"""

import os
import xml.dom.minidom
import re
import csv
import pydicom as dicom


def saveCoord(docter_id,elems3,xCoords,yCoords,dcm_Name,imageSOP_UID):
    with open('allxyCoord_test.csv','a+',newline='') as csvfile:
        writer = csv.writer(csvfile)
        for x,y in zip(xCoords,yCoords):
            writer.writerow([docter_id]+elems3+[x.childNodes[0].data,y.childNodes[0].data,dcm_Name,imageSOP_UID])
    csvfile.close()

def saveAll2csv(elems):
    with open('xmlKeydata_test.csv','a+',newline='') as csvf:
        writer =csv.writer(csvf)
        writer.writerow(elems)
    csvf.close()

def findROIuid(xml_SOP_UID,file_path):
    dcm_list = os.listdir(file_path)
    for dcm in dcm_list:
        try:
            slices = dicom.read_file(file_path + dcm)
            SOP_UID = slices.SOPInstanceUID
            if xml_SOP_UID == SOP_UID:
                return dcm
        except:
            pass

def recordNodule_Non(e,elems1to2,docter_id):
    # unblindedReadNodule
    unblindedReadNodule = e.getElementsByTagName('unblindedReadNodule')
    for u in unblindedReadNodule:
        elems3,elems4to12= [],[]
        elems3.append('unblindedReadNodule')
        noduleID = u.getElementsByTagName('noduleID')[0].childNodes[0].data
        elems3.append(noduleID)
        try:
            for i in nodes4to12:
                elems4to12.append(u.getElementsByTagName(i)[0].childNodes[0].data)
            elems3.append('>=3mm')
        except:
            for j in range(9):
                elems4to12.append('')
            elems3.append('<3mm')
        # roi
        roi_lists = u.getElementsByTagName('roi')
        for r_list in roi_lists:
            elems13to15 = []
            imageZposition = r_list.getElementsByTagName('imageZposition')[0].childNodes[0].data  # slice
            elems13to15.append(imageZposition)
            imageSOP_UID = r_list.getElementsByTagName('imageSOP_UID')[0].childNodes[0].data  # slice id
            elems13to15.append(imageSOP_UID)
            dcm_par_path = su_file[:-7]
            elems13to15.append(dcm_par_path)
            saveAll2csv(elems1to2 + elems3 + elems4to12 + elems13to15)
            # Find the corresponding dcm file to the slice
            dcm_Nmae = findROIuid(imageSOP_UID, dcm_par_path)
            xCoords = r_list.getElementsByTagName('xCoord')
            yCoords = r_list.getElementsByTagName('yCoord')
            saveCoord(docter_id,elems3,xCoords, yCoords, dcm_Nmae, imageSOP_UID)
    # nonNodule
    nonNodule = e.getElementsByTagName('nonNodule')
    for n in nonNodule:
        elems3 = []
        elems4to12 = []
        elems3.append('nonNodule')
        noduleID = n.getElementsByTagName('nonNoduleID')[0].childNodes[0].data
        elems3.append(noduleID)
        elems3.append('')
        for k in range(9):
            elems4to12.append('')
        elems13to15 = []
        imageZposition = n.getElementsByTagName('imageZposition')[0].childNodes[0].data  # slice
        elems13to15.append(imageZposition)
        imageSOP_UID = n.getElementsByTagName('imageSOP_UID')[0].childNodes[0].data  # slice id
        elems13to15.append(imageSOP_UID)
        dcm_par_path = su_file[:-7]
        elems13to15.append(dcm_par_path)
        saveAll2csv(elems1to2 + elems3 + elems4to12 + elems13to15)
        # Find the corresponding dcm file to the slice
        dcm_Nmae = findROIuid(imageSOP_UID, dcm_par_path)
        xCoords = n.getElementsByTagName('xCoord')
        yCoords = n.getElementsByTagName('yCoord')
        saveCoord(docter_id,elems3,xCoords, yCoords, dcm_Nmae, imageSOP_UID)

def readXml(su_file):
    xml_file = xml.dom.minidom.parse(su_file)
    # 保存xml所有关键信息
    root = xml_file.documentElement
    readingSession = root.getElementsByTagName('readingSession')
    elems1to2 = []
    for e in readingSession:
        file_Name = su_file[29:43]
        elems1to2.append(file_Name)
        docter_id = e.getElementsByTagName('servicingRadiologistID')[0].childNodes[0].data
        elems1to2.append(docter_id)
        recordNodule_Non(e, elems1to2,docter_id)
        elems1to2 = []




if __name__=='__main__':
    # root_path = '../../Medical_Data/LIDC-IDRI'
    #---test file---
    root_path = 'LIDC-IDRI'
    #----------
    nodes4to12 = ['subtlety','internalStructure','calcification','sphericity',
                  'margin','lobulation','spiculation','texture','malignancy']
    for dirpath, dirnames, filenames in os.walk(root_path):
        if len(filenames) > 10:
            for filepath in filenames:
                su_file = os.path.join(dirpath, filepath)
                print(su_file)
                try:
                    readXml(su_file)
                except:
                    pass
    print('finish')