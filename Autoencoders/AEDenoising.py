
# coding: utf-8

# In[1]:

import numpy as np
import time


# In[5]:

import sys
sys.path.append('../FileOps/')
import PatchSample


# In[3]:

# denoising (not used for the recon)
def SAEDenoising(img, sae, sess, img_latent = None,
                 strides = None, batchsize=1000, step=0.2, nSteps=10, calcLoss=False, random=True):
    if strides is None:
        strides = [sae.imgshape[0] / 2, sae.imgshape[1] / 2]
    
    if img_latent is None:
        img_latent = np.copy(img)
    
    if random:
        patches, x, y = PatchSample.MakePatches2DRandom(img, [sae.imgshape[0], sae.imgshape[1]], strides)
    else:
        patches, x, y = PatchSample.MakePatches2D(img, [sae.imgshape[0], sae.imgshape[1]], strides)
    patches_latent,_,_ = PatchSample.MakePatches2DRandom(img_latent, [sae.imgshape[0], sae.imgshape[1]], strides, x, y)

    for i in range(nSteps):
        grads = sae.GetRefGradients(patches_latent, patches, batchsize, sess)
        for j in range(grads.shape[0]):
            grads[j,...] = grads[j,...] / np.linalg.norm(grads[j,...])
        latent_patches = patches_latent - step * grads
        if calcLoss:
            loss = sae.GetRefLoss(patches_latent, patches, batchsize, sess)
            print loss
    
    denoised_patches = sae.Predict(patches_latent, batchsize, sess)
    
    kernel = PatchSample.GetGaussianKernel([sae.imgshape[0], sae.imgshape[1]], sae.imgshape[0]/3.0)
    denoised_img = PatchSample.AggregatePatches2D(denoised_patches, x, y, img.shape, kernel)
    
    return denoised_img


# In[8]:

# SQS step of the penalty
# img: image to process
# sae: the sparse autoencoder
# sess: the session to run
# img_latent: has to be None
# patches_latent: should be None
def SAEDenoisingSQS(img, sae, sess, img_latent=None, patches_latent = None,
                    strides = None, batchsize=1000, step=0.05, nSteps=10, calcLoss=False, random=True):
    if strides is None:
        strides = [sae.imgshape[0] / 2, sae.imgshape[1] / 2]
    if img_latent is None:
        img_latent = np.copy(img)
    
    if random:
        patches, x, y = PatchSample.MakePatches2DRandom(img, [sae.imgshape[0], sae.imgshape[1]], strides)
    else:
        patches, x, y = PatchSample.MakePatches2D(img, [sae.imgshape[0], sae.imgshape[1]], strides)
    if patches_latent is None:
        patches_latent,_, _ = PatchSample.MakePatches2DRandom(img_latent, [sae.imgshape[0], sae.imgshape[1]], 
                                                              strides, x, y)

    for i in range(nSteps):
        grads = sae.GetRefGradients(patches_latent, patches, batchsize, sess)
        for j in range(grads.shape[0]):
            grads[j,...] = grads[j,...] / np.linalg.norm(grads[j,...])
        patches_latent = patches_latent - step * grads
        if calcLoss:
            loss = sae.GetRefLoss(patches_latent, patches, batchsize, sess)
            print loss
    denoised_patches = sae.Predict(patches_latent, batchsize, sess)
    
    kernel = np.ones([sae.imgshape[0], sae.imgshape[1]])
    cf = np.sum((patches - denoised_patches)**2)
    sum_difference = PatchSample.SumPatches2D(patches - denoised_patches, x, y, img.shape, kernel)
    sum_ones = PatchSample.SumPatches2D(np.ones(patches.shape), x, y, img.shape, kernel)

    return sum_difference, sum_ones, cf, patches_latent


# In[7]:

# SAE SQS step for 3D
def SAEDenoisingSQS3D(img, sae, sess, strides = None, batchsize=1000, step=0.1, nSteps=1, axis=0, output=True):
    if strides is None:
        strides = [sae.imgshape[0] / 2, sae.imgshape[1] / 2]
    patchsize = [sae.imgshape[0], sae.imgshape[1]]
    if axis != 0 and axis != 1 and axis != 2:
        raise ValueError('axis must be one of 1, 2, 3')
    
    layer_zero = np.take(img, 0, axis)
    x,y = PatchSample.GetPatchesCoordsRandom(layer_zero, patchsize, strides)
    
    patches = np.ones([len(x)*len(y), patchsize[0], patchsize[1]], np.float32)
    
    # get the normalize image
    kernel = np.ones([sae.imgshape[0], sae.imgshape[1]])
    sum_ones = PatchSample.SumPatches2D(np.ones(patches.shape), x, y, img.shape, kernel)
    
    sum_ones = np.repeat(np.expand_dims(sum_ones, axis), img.shape[axis], axis)
    
    sum_difference = np.zeros(img.shape, np.float32)
    cf = 0
    output_interval = img.shape[axis] / 20
    for iLayer in range(img.shape[axis]):
        if output and output_interval > 0:
            if (iLayer + 1) % output_interval == 0:
                print '%d...'%iLayer,
        
        imgLayer = np.copy(np.take(img, iLayer, axis))
        # extract patches
        FastPatch.ImgToPatchesWithCoords(patches, imgLayer, x, y)
        patches_latent = np.copy(patches)
        
        # gradient descend
        start = time.time()
        for iStep in range(nSteps):
            grads = sae.GetRefGradients(patches_latent, patches, batchsize, sess)
            for j in range(grads.shape[0]):
                grads[j,...] = grads[j,...] / np.linalg.norm(grads[j,...])
            patches_latent = patches_latent - step * grads
        denoised_patches = sae.Predict(patches_latent, batchsize, sess)
        
        # put the errors back
        sumDiffLayer = np.zeros(imgLayer.shape, np.float32)
        FastPatch.PatchesToImgWithCoordsSQS(sumDiffLayer, patches - denoised_patches, x, y)
        if axis == 0:
            sum_difference[iLayer, :, :] = sumDiffLayer
        elif axis == 1:
            sum_difference[:, iLayer, :] = sumDiffLayer
        elif axis == 2:
            sum_difference[:, :, iLayer] = sumDiffLayer
        cf += np.sum((patches - denoised_patches)**2)
    
    return sum_difference, sum_ones, cf
    

