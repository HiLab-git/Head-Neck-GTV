"""Script for preprocess
"""
import os
import numpy as np
import shutil


def preprocess_gtv(dataroot,saveroot):
    # 移动子文件夹内的data&label至一个文件夹
    for name in os.listdir(dataroot):
        fileroot = dataroot+name
        newroot = saveroot+'data'
        segroot = saveroot+'label'
        odpath = fileroot+'/'+filename
        osegpath = fileroot+'/'+segname
        newfilename = name + filename
        newsegname = newfilename.replace('ta', 'ta_seg')
        ndpath = newroot+'/'+ newfilename
        nsegpath = segroot+'/'+ newsegname
        shutil.copy(odpath,ndpath)
        shutil.copy(osegpath,nsegpath)

if __name__ == "__main__":
    dataroot = ''   #write your own dataroot here
    saveroot = ''   #write your own saveroot here
    filename = 'data.nii.gz'
    segname = 'label.nii.gz'
    preprocess_gtv(dataroot,saveroot)