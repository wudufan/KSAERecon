
# coding: utf-8

# In[2]:

from glob import glob
import numpy as np
import os
import sys
import dicom
import FileIO
import h5py
import scipy.signal


# In[2]:

# generate samples from ct_img in certain orientation
# ct_img: the image in z,y,x order
# sampleshape: batchsize * w * h * c
# orientation: e.g. xyz means the batches are sampled along x,y,z; xzy means the batches are sampled along x,z,y
# mean_along_channels: if the output samples are meaned along the channels
# coords: the sample coordiante for the patches. If none, they are sampled randomly
def GenerateSamples(ct_img, sampleshape=[1000,64,64,1], orientation='xyz', mean_along_channels=False, coords=None):
    orientation = orientation.lower()
    if orientation == 'xyz':
        patchshape = [sampleshape[1], sampleshape[2], sampleshape[3]]
    elif orientation == 'xzy':
        patchshape = [sampleshape[1], sampleshape[3], sampleshape[2]]
    elif orientation == 'yxz':
        patchshape = [sampleshape[2], sampleshape[1], sampleshape[3]]
    elif orientation == 'yzx':
        patchshape = [sampleshape[3], sampleshape[1], sampleshape[2]]
    elif orientation == 'zxy':
        patchshape = [sampleshape[2], sampleshape[3], sampleshape[1]]
    elif orientation == 'zyx':
        patchshape = [sampleshape[3], sampleshape[2], sampleshape[1]]
    else:
        raise ValueError('orientation should be one of: xyz, xzy, yxz, yzx, zxy, zyx')
    
    axes_order = ['zyx'.index(orientation[0]), 'zyx'.index(orientation[1]), 'zyx'.index(orientation[2])]
    
    if coords is None:
        x = np.random.randint(0, ct_img.shape[-1] - patchshape[0] + 1, sampleshape[0])
        y = np.random.randint(0, ct_img.shape[-2] - patchshape[1] + 1, sampleshape[0])
        z = np.random.randint(0, ct_img.shape[-3] - patchshape[2] + 1, sampleshape[0])
    else:
        x = coords[0]
        y = coords[1]
        z = coords[2]
    
    samples = np.zeros(sampleshape, dtype=np.float32)
    for i in range(sampleshape[0]):
        samples[i,...] = np.transpose(ct_img[z[i]:z[i]+patchshape[2], y[i]:y[i]+patchshape[1], x[i]:x[i]+patchshape[0]],
                                      axes_order)
    
    if mean_along_channels:
        samples = np.mean(samples, -1)[..., np.newaxis]
    
    return samples, (x,y,z)
    


# In[5]:

# read a dicom sequence and generate training patches, and save
def GenerateTrainingPatchesFromDicomSeq(folder, nFile, nImgPerFile, nSamplePerImg, imgshape, 
                                        dicomSeqList, dicomSeqPostfix='*.IMA', 
                                        orientationList = ['xyz', 'xzy', 'yxz', 'yzx', 'zxy', 'zyx'],
                                        overwrite=False, normFac=500.0, mean_along_channels = False):
    if not os.path.exists(folder):
        os.makedirs(folder)
    elif not overwrite:
        print 'Training patches already exist at folder: %s'%folder
        return [nFile, nImgPerFile * nSamplePerImg] + imgshape
    
    for iFile in range(nFile):
        print 'Generating file %d of %d'%(iFile+1, nFile)
        data = np.zeros([nImgPerFile * nSamplePerImg] + imgshape, dtype=np.float32)
        for iImg in range(nImgPerFile):
            img, _ = FileIO.ReadFromDicomSeq(dicomSeqList[np.random.randint(len(dicomSeqList))], dicomSeqPostfix)
            samples,_ = GenerateSamples(img, [nSamplePerImg] + imgshape, 
                                        orientationList[np.random.randint(len(orientationList))])
            samples /= normFac
            data[iImg*nSamplePerImg:(iImg+1)*nSamplePerImg, ...] = samples
        if mean_along_channels:
            data = np.mean(data, -1)[..., np.newaxis]
        with h5py.File(os.path.join(folder, '%d.h5'%iFile), 'w') as f:
            f['data'] = data.astype(np.float32)
    
    return [nFile] + list(data.shape)
    


# In[6]:

# read from h5 files
def RetrieveTrainingPatches(folder, iFile):
    with h5py.File(os.path.join(folder, '%d.h5'%iFile), 'r') as f:
        data = np.copy(f['data'])
    
    return data


# In[24]:

# make patches from 2D images on grids 
def MakePatches2D(img, patchsize, strides):
    x = range(0, img.shape[0]-patchsize[0]+1, strides[0])
    y = range(0, img.shape[1]-patchsize[1]+1, strides[1])
    x.append(img.shape[0]-patchsize[0])
    y.append(img.shape[1]-patchsize[1])
    
    patches = np.zeros([len(x) * len(y)] + patchsize)
    for ix in range(len(x)):
        for iy in range(len(y)):
            patches[ix * len(y) + iy] = img[x[ix]:x[ix]+patchsize[0], y[iy]:y[iy]+patchsize[1]]
    
    return patches, x, y


# In[ ]:

# make patches 2D on random coordinates. pass coordinates from x and y, if not given, the grids were perturbed randomly
def MakePatches2DRandom(img, patchsize, strides, x=None, y=None):
    
    if x is None:
        x = range(0, img.shape[0]-patchsize[0]+1, strides[0])
        x.append(img.shape[0]-patchsize[0])
        x = np.asarray(x)
        x[1:-1] += np.random.randint((strides[0] - patchsize[0]+1) / 2, (patchsize[0] - strides[0]) / 2, [len(x)-2])
        x[x > img.shape[0] - patchsize[0]] = img.shape[0]-patchsize[0]
        
    if y is None:
        y = range(0, img.shape[1]-patchsize[1]+1, strides[1])
        y.append(img.shape[1]-patchsize[1])
        y = np.asarray(y)
        y[1:-1] += np.random.randint((strides[1] - patchsize[1]+1) / 2, (patchsize[1] - strides[1]) / 2, [len(y)-2])
        y[y > img.shape[1] - patchsize[1]] = img.shape[1]-patchsize[1]

    patches = np.zeros([len(x) * len(y)] + patchsize)
    for ix in range(len(x)):
        for iy in range(len(y)):
            patches[ix * len(y) + iy] = img[x[ix]:x[ix]+patchsize[0], y[iy]:y[iy]+patchsize[1]]
    
    return patches, x, y


# In[1]:

# generate the coordinates of patches in 2D
# (for arregation using fast patch module)
def GetPatchesCoordsRandom(img, patchsize, strides):
    x = range(0, img.shape[-1]-patchsize[-1]+1, strides[-1])
    x.append(img.shape[-1]-patchsize[-1])
    x = np.asarray(x)
    x[1:-1] += np.random.randint((strides[-1] - patchsize[-1]+1) / 2, (patchsize[-1] - strides[-1]) / 2, [len(x)-2])
    x[x > img.shape[-1] - patchsize[-1]] = img.shape[-1]-patchsize[-1]
    
    y = range(0, img.shape[-2]-patchsize[-2]+1, strides[-2])
    y.append(img.shape[-2]-patchsize[-2])
    y = np.asarray(y)
    y[1:-1] += np.random.randint((strides[-2] - patchsize[-2]+1) / 2, (patchsize[-2] - strides[-2]) / 2, [len(y)-2])
    y[y > img.shape[-2] - patchsize[-2]] = img.shape[-2]-patchsize[-2]
    
    return x.astype(np.int32),y.astype(np.int32)
    


# In[3]:

# generate 2D patches along axis
def MakePatches2DRandomFrom3D(img, patchsize, strides, x=None, y=None, main_axis='z'):
    # image as zyx
    working_img = np.copy(img)
    if main_axis == 'y' or main_axis == 1:
        working_img = np.swapaxes(working_img, 0, 1)
    elif main_axis == 'x' or main_axis == 2:
        working_img = np.swapaxes(working_img, 0, 2)
    
    if x is None:
        x = range(0, img.shape[2]-patchsize[2]+1, strides[2])
        x.append(img.shape[2]-patchsize[2])
        x = np.asarray(x)
        x[1:-1] += np.random.randint((strides[2] - patchsize[2]+1) / 2, (patchsize[2] - strides[2]) / 2, [len(x)-2])
        x[x > img.shape[2] - patchsize[2]] = img.shape[2]-patchsize[2]
        
    if y is None:
        y = range(0, img.shape[1]-patchsize[1]+1, strides[1])
        y.append(img.shape[1]-patchsize[1])
        y = np.asarray(y)
        y[1:-1] += np.random.randint((strides[1] - patchsize[1]+1) / 2, (patchsize[1] - strides[1]) / 2, [len(y)-2])
        y[y > img.shape[1] - patchsize[1]] = img.shape[1]-patchsize[1]
    
    patches = np.zeros([len(x), len(y), working_img.shape[0], patchsize[1], patchsize[2]], np.float32)
    for ix in range(len(x)):
        for iy in range(len(y)):
            patches[ix, iy, ...] = working_img[:, y[iy]:y[iy]+patchsize[1], x[ix]:x[ix]+patchsize[2]]
    
    return patches, x, y
    
    


# In[27]:

# aggregate patches with normalization
def AggregatePatches2D(patches, x, y, imgshape, weights):
    img = np.zeros(imgshape, np.float32)
    imgw = np.zeros(imgshape, np.float32)
    patchsize = [patches.shape[1], patches.shape[2]]
    
    for ix in range(len(x)):
        for iy in range(len(y)):
            img[x[ix]:x[ix]+patchsize[0], y[iy]:y[iy]+patchsize[1]] += patches[ix*len(y) + iy] * weights
            imgw[x[ix]:x[ix]+patchsize[0], y[iy]:y[iy]+patchsize[1]] += weights
    
    return img / imgw
    


# In[ ]:

# sum patches without normalization (for SQS)
def SumPatches2D(patches, x, y, imgshape, weights):
    img = np.zeros(imgshape, np.float32)
    patchsize = [patches.shape[1], patches.shape[2]]
    
    for ix in range(len(x)):
        for iy in range(len(y)):
            img[x[ix]:x[ix]+patchsize[0], y[iy]:y[iy]+patchsize[1]] += patches[ix*len(y) + iy] * weights
    
    return img


# In[2]:

def SumPatches2DTo3D(patches, x, y, imgshape, main_axis='z'):
    if main_axis == 'z' or main_axis == 0:
        working_shape = [imgshape[0], imgshape[1], imgshape[2]]
    elif main_axis == 'y' or main_axis == 1:
        working_shape = [imgshape[1], imgshape[0], imgshape[2]]
    elif main_axis == 'x' or main_axis == 2:
        working_shape = [imgshape[2], imgshape[1], imgshape[0]]
    
    img = np.zeros(working_shape, np.float32)
    patchsize = [1, patches.shape[-2], patches.shape[-1]]
    
    for ix in range(len(x)):
        for iy in range(len(y)):
            img[:, y[iy]:y[iy]+patchsize[1], x[ix]:x[ix]+patchsize[2]] += patches[ix, iy, ...]
    
    if main_axis == 'y' or main_axis == 1:
        img = np.swapaxes(img, 0, 1)
    elif main_axis == 'x' or main_axis == 2:
        img = np.swapaxes(img, 0, 2)
    
    return img


# In[33]:

def GetGaussianKernel(patchsize, std):
    h1 = scipy.signal.gaussian(patchsize[0], std)[...,np.newaxis]
    h2 = scipy.signal.gaussian(patchsize[1], std)[np.newaxis,...]
    
    h = np.dot(h1,h2)
    return h / h.sum()


# In[ ]:



