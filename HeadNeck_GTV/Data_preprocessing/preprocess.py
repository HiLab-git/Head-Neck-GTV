
from __future__ import absolute_import, print_function
import nibabel
import numpy as np
import os.path
import os
from scipy import ndimage as ndi
from scipy import ndimage
from skimage.measure import label as lb
from skimage.measure import regionprops
from skimage import morphology
from skimage.filters import roberts

def save_array_as_nifty_volume(data, filename, transpose=True):
    """
    save a numpy array as nifty image
    inputs:
        data: a numpy array with shape [Channel, Depth, Height, Width]
        filename: the ouput file name
    outputs: None
    """
    if transpose:
        data = data.transpose(2, 1, 0)
    img = nibabel.Nifti1Image(data, None)
    nibabel.save(img, filename)


def load_origin_nifty_volume_as_array(filename):
    """
    load nifty image into numpy array, and transpose it based on the [z,y,x] axis order
    The output array shape is like [Depth, Height, Width]
    inputs:
        filename: the input file name, should be *.nii or *.nii.gz
    outputs:
        data: a numpy data array
        zoomfactor:
    """
    img = nibabel.load(filename)
    pixelspacing = img.header.get_zooms()
    zoomfactor = list(pixelspacing)
    zoomfactor.reverse()
    data = img.get_data()
    data = data.transpose(2, 1, 0)
#     print(data.shape)

    return data, zoomfactor

def zoom_data(file, mode='img', zoom_factor=[1,1,1], class_number=0):
    """
    对数据进行插值并储存，
    :param data_root: 数据所在上层目录
    :param save_root: 存储的顶层目录
    :zoom_factor:   缩放倍数
    :return:
    """

    if mode =='label':
        intfile = np.int16(file)
        #zoom_file = np.int16(resize_Multi_label_to_given_shape(intfile, zoom_factor, class_number, order=2))
        zoom_file = ndimage.interpolation.zoom(file, zoom_factor, order=0)
    elif mode == 'img':
        zoom_file = ndimage.interpolation.zoom(file, zoom_factor, order=1)
    else:
        KeyError('please choose img or label mode')
    return zoom_file

def crop(file, bound):
    '''
    :param file: z, x, y
    :param bound: [min,max]
    :return:
    '''

    cropfile = file[max(bound[0][0], 0):min(bound[1][0], file.shape[0]), bound[0][1]:bound[1][1], bound[0][2]:bound[1][2]]
    return cropfile

def img_normalized(file, upthresh=0, downthresh=0, norm=True, thresh=True):
    """
    :param file: np array
    :param upthresh:
    :param downthresh:
    :param norm: norm or not
    :return:
    """
    if thresh:
        assert upthresh > downthresh
        file[np.where(file > upthresh)] = upthresh
        file[np.where(file < downthresh)] = downthresh
    if norm:
        file = (file-downthresh)/(upthresh-downthresh)
    return file

def get_bound_coordinate(file, pad=[0,0,0]):
    '''
    输出array非0区域的各维度上下界坐标+-pad
    :param file: groundtruth图,
    :param pad: 各维度扩充的大小
    :return: bound: [min,max]
    '''
    nonzeropoint = np.asarray(np.nonzero(file))   # 得到非0点坐标,输出为一个3*n的array，3代表3个维度，n代表n个非0点在对应维度上的坐标
    maxpoint = np.max(nonzeropoint, 1).tolist()
    minpoint = np.min(nonzeropoint, 1).tolist()
    for i in range(len(pad)):
        maxpoint[i] = maxpoint[i]+pad[i]
        minpoint[i] = minpoint[i]-pad[i]
    return [minpoint, maxpoint]

def get_segmented_body1(img, window_max=250, window_min=-150, window_length=0, show_body=False, znumber=0):
    '''
    将身体与外部分离出来
    '''

    mask = []

    if znumber < 40:
        radius = [13, 6]
    else:
        radius = [6, 8]


    '''
    Step 1: Convert into a binary image.二值化,为确保所定阈值通过大多数
    '''
    threshold = -600
    binary = np.where(img > threshold, 1.0, 0.0)  # threshold the image

    '''
    Step 2: Remove the blobs connected to the border of the image.
            清除边界
    '''

    '''
    Step 3: Erosion operation with a disk of radius 2. This operation is
    seperate the lung nodules attached to the blood vessels.
    腐蚀操作，以2mm为半径去除
    '''
    binary = morphology.erosion(binary, np.ones([radius[0], radius[0]]))

    '''
    Step 4: Closure operation with a disk of radius 10. This operation is
    to keep nodules attached to the lung wall.闭合运算
    '''
    binary = morphology.dilation(binary, np.ones([radius[1], radius[1]]))
    '''
    Step 5: Label the image.连通区域标记
    '''
    label_image = lb(binary)

    '''
    Step 6: Keep the labels with the largest area.保留最大区域
    '''
    areas = [r.area for r in regionprops(label_image)]
    areas.sort()
    if len(areas) > 1:
        for region in regionprops(label_image):
            if region.area < areas[-1]:
                for coordinates in region.coords:
                    label_image[coordinates[0], coordinates[1]] = 0
    binary = label_image > 0

    '''
    Step 7: Fill in the small holes inside the binary mask .孔洞填充
    '''
    edges = roberts(binary)
    binary = ndi.binary_fill_holes(edges)
    '''
    Step 8: show the input image.
    '''

    '''
    Step 9: Superimpose the binary mask on the input image.
    '''
    get_high_vals = binary == 0
    img[get_high_vals] = 0
    mask.append(binary)

    img[img > (window_max + window_length)] = window_max + window_length
    img[img < (window_min - window_length)] = window_min - window_length
    img = (img - window_min) / (window_max - window_min)
    img[get_high_vals] = 0

    return img, binary

def preprocessing(img,label, pixelspace,pad=[0,0,0],cropmode='small'):
    '''
    data preprocessing
    '''

    zoomfactor = [pixelspace[0]/3, pixelspace[1], pixelspace[2]]
    img = zoom_data(img, mode='img', zoom_factor=zoomfactor)
    label = zoom_data(label, mode='label', zoom_factor=zoomfactor)
    ishape = np.copy(img)
    ishape = ishape.shape
    a = np.copy(img)
    np.where(a < -900,a,0)
    e,_ = get_segmented_body1(a[0])
    e = np.expand_dims(e,axis=0)
    for i in range(1,a.shape[0]):
        c,_ = get_segmented_body1(a[i])
        d = np.expand_dims(c,axis=0)
        e = np.concatenate((e,d), 0)

    bound = get_bound_coordinate(e, pad=pad)# [min,max]
    newbound = [[],[]]
    if cropmode == 'small':
        newbound[0] = [bound[0][0]+50,bound[0][1]+(bound[1][1]-bound[1][0])//10,bound[0][2]+(bound[1][2]-bound[0][2])//3+5]
        newbound[1] = [bound[1][0],bound[1][1]-(bound[1][1]-bound[1][0])//3,bound[1][2]-(bound[1][2]-bound[0][2])//3-5]
    elif cropmode == 'middle':
        newbound[0] = [bound[0][0]+35,bound[0][1]+(bound[1][1]-bound[1][0])//10-5,bound[0][2]+(bound[1][2]-bound[0][2]-5)//3+5]
        newbound[1] = [bound[1][0],bound[1][1]-(bound[1][1]-bound[1][0])//3+10,bound[1][2]-(bound[1][2]-bound[0][2])//3]
    else:
        newbound[0] = [bound[0][0]+16,bound[0][1],bound[0][2]]
        newbound[1] = [bound[1][0],bound[1][1],bound[1][2]]
    newbound = [newbound[0], newbound[1]]

    cropimg = crop(img, newbound)
    cropimg = img_normalized(cropimg, upthresh=700, downthresh=-200, norm=True, thresh=True)
    croplabel = crop(label, newbound)

    if cropimg.shape[0] % 16 != 0:
        i = cropimg.shape[0] // 16
        i = i * 16
        cropimg = cropimg[:i,:,:]
        croplabel = croplabel[:i,:,:]

    if cropimg.shape[1] % 16 != 0:
        i = cropimg.shape[1] // 16
        i = i * 16
        cropimg = cropimg[:,:i,:]
        croplabel = croplabel[:,:i,:]

    if cropimg.shape[2] % 16 != 0:
        i = cropimg.shape[2] // 16
        i = i * 16
        cropimg = cropimg[:,:,:i]
        croplabel = croplabel[:,:,:i]

    return cropimg, croplabel

if __name__ == '__main__':
    '''
    Put your own path here
    '''
    imgroot = '/home/uestcc1501h/alldataset/origindata/data'
    labelroot = '/home/uestcc1501h/alldataset/origindata/label'

    saveroot_small = '/home/uestcc1501h/alldataset/small_scale/'
    smalldatanewroot = saveroot_small+'data'
    smalllabelnewroot = saveroot_small+'label'

    saveroot_middle = '/home/uestcc1501h/alldataset/middle_scale/'
    middledatanewroot = saveroot_middle+'data'
    middlelabelnewroot = saveroot_middle+'label'

    saveroot_large = '/home/uestcc1501h/alldataset/large_scale/'
    largedatanewroot = saveroot_large+'data'
    largelabelnewroot = saveroot_large+'label'


    for imgname in os.listdir(imgroot):
        print('imgname is ',imgname)
        '''
        根据原label得到新label/img的名称,像素间距与储存路径
        '''
        labelname = imgname.replace('ta', 'ta_seg')
        imgpath = os.path.join(imgroot, imgname)
        labelpath = os.path.join(labelroot, labelname)
        imgnewpath_small = os.path.join(smalldatanewroot, imgname)
        labelnewpath_small = os.path.join(smalllabelnewroot, labelname)

        imgnewpath_middle = os.path.join(middledatanewroot, imgname)
        labelnewpath_middle = os.path.join(middlelabelnewroot, labelname)

        imgnewpath_large = os.path.join(largedatanewroot, imgname)
        labelnewpath_large = os.path.join(largelabelnewroot, labelname)

        img, pixelspace = load_origin_nifty_volume_as_array(imgpath)
        label, _ = load_origin_nifty_volume_as_array(labelpath)

        cropdata_small, crop_label_small = preprocessing(img,label, pixelspace, cropmode = 'small')
        save_array_as_nifty_volume(crop_label_small, labelnewpath_small)
        save_array_as_nifty_volume(cropdata_small,imgnewpath_small)

        cropdata_middle, crop_label_middle = preprocessing(img,label, pixelspace, cropmode = 'middle')
        save_array_as_nifty_volume(crop_label_middle, labelnewpath_middle)
        save_array_as_nifty_volume(cropdata_middle,imgnewpath_middle)

        cropdata_large, crop_label_large = preprocessing(img,label, pixelspace, cropmode = 'large')
        save_array_as_nifty_volume(crop_label_large, labelnewpath_large)
        save_array_as_nifty_volume(cropdata_large,imgnewpath_large)