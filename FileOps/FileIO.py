
# coding: utf-8

# In[1]:

from glob import glob
import numpy as np
import os
import sys
import dicom


# In[2]:

def ReadFromDicomSeq(path, postfix='*.dcm'):
    imgList = list()
    fileNames = glob(os.path.join(path, postfix))
    zPos = list()
    for filename in fileNames:
        d = dicom.read_file(filename, force=True)
        zPos.append(d.ImagePositionPatient[2])
#         imgList.append(d.pixel_array.astype(np.float32))
        imgList.append(d.pixel_array.astype(np.float32) * d.RescaleSlope + d.RescaleIntercept)

    indices = np.argsort(np.asarray(zPos))
    indices = indices[::-1]
    zPos = [zPos[i] for i in indices]
    imgList = [ imgList[i] for i in indices]
    img = np.asarray(imgList)
    
    info = {}
    info['zPos'] = zPos
    info['dx'] = d.PixelSpacing[0]
    info['dz'] = abs(zPos[0] - zPos[1])
    
    return img, info


# In[ ]:



