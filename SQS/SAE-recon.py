
# coding: utf-8

# In[ ]:

get_ipython().magic(u'matplotlib inline')

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import time
import tensorflow as tf
from IPython import display
import scipy.signal


# In[ ]:

sys.path.append('../pythonWrapper/')
import EAProjectorWrapper

sys.path.append('../FileOps/')
import PatchSample
import FileIO

sys.path.append('../Autoencoders/')
import AEDenoising
import SSAE


# In[ ]:

dataPath = '/home/data1/dufan/lowdoseCTsets/L291/'
prj = EAProjectorWrapper.Projector()
prj.FromFile(os.path.join(dataPath, 'param.txt'))

layer = 78 #L291
with open(os.path.join(dataPath, 'quarter.raw'), 'rb') as f:
    f.seek(prj.nu*prj.rotview*layer*4, os.SEEK_SET)
    sino = np.fromfile(f, dtype=np.float32, count=prj.nu*prj.rotview)
    f.close()
sino = sino.reshape([prj.rotview, prj.nu])


# In[ ]:

img0 = np.fromfile('/home/data0/dufan/Reconstruction/recon_new/recon/L291-78/fbp-quarter-3mean.raw',  dtype=np.float32)
img0 = (img0 + 1000) / 1000 * 0.01937
img0 = np.reshape(img0, [640,640])

plt.figure(figsize=[8,8])
plt.imshow(img0 / 0.01937 * 1000 - 1000, 'Greys_r', vmin=-160, vmax=240)


# In[ ]:

def SAEReconSQS(sino, img0, prj, sae, sess, strides = None, nIter = 1,
                hyper=0, subStepSize=0.05, nSubSteps=5, gamma = 0.5, random_patch = True, showSAELoss = False):
    if strides is None:
        strides = [sae.imgshape[0] / 2, sae.imgshape[1] / 2]
    
    # pre calculation
    # w = sqrt(exp(-sino)) / prj_ones gives more stable results than exp(-sino) weighting
    prj_ones = prj.ProjectionEA(np.ones(img0.shape, dtype=np.float32)) + 1e-6
    w = np.sqrt(np.exp(-sino)) / prj_ones
    normImg = prj.BackProjectionEA(w * prj.ProjectionEA(np.ones(img0.shape, dtype=np.float32)))
        
    total_cfs = list()
    sae_cfs = list()
    x = np.copy(img0)
    z = np.copy(x)
    patches_latent = None
    for iIter in range(nIter):
        x_input = x / 0.01937 * 2 - 2
        y_input = np.copy(x_input)
        sum_diff, sum_ones, cf_sae, _ =             AEDenoising.SAEDenoisingSQS(x_input, sae, sess, y_input, None,
                                        strides, step=subStepSize, nSteps = nSubSteps,
                                        random=random_patch, calcLoss=showSAELoss)

        sum_diff = sum_diff / 2 * 0.01937
        cf_sae = cf_sae / 2 / 2 * 0.01937 * 0.01937 
        
        dprj = (prj.ProjectionEA(x) - sino)
        dprj[prj_ones <= 1e-6] = 0
        dimg_prj = prj.BackProjectionEA(w * dprj)
        
        # Nesterov Momentum
        x_new = z - (dimg_prj + 2 * hyper * sum_diff) / (normImg + 2 * hyper * sum_ones)
        z = x_new + gamma * (x_new - x)
        x = np.copy(x_new)
                
        cf_prj = 0.5 * np.sum(w * dprj**2)
        cf = cf_prj + hyper * cf_sae
        total_cfs.append(cf)
        sae_cfs.append(cf_sae)
        
        display.clear_output()
        print 'CF=(%f, %f, %f)'%(cf, cf_prj, cf_sae)
        plt.figure(figsize=[16,8])
        plt.subplot(121); plt.imshow(x / 0.01937 * 1000 - 1000, 'Greys_r', vmin=-160, vmax=240); plt.title('Image at %d'%iIter)
        plt.subplot(222); plt.plot(sae_cfs); plt.xlim((0, nIter)); plt.title('SAE loss')
        plt.subplot(224); plt.semilogy(total_cfs); plt.xlim((0, nIter)); plt.title('Total loss')
        plt.show()        
    
    return x, total_cfs, sae_cfs


# In[ ]:

sparsity = 100
sparsity_src = 100
tf.reset_default_graph()
ae = SSAE.StackedSparseAutoEncoder(imgshape=[16,16,1], nFeatures=[1024,1024,1024], 
                                   sparsity=[sparsity,sparsity,sparsity], mode=0)
ae.BuildStackedAutoEncoder(scope='SSAE')
ae.BuildGradientsWRTInput(scope='SSAE')
sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(visible_device_list='0', 
                                                                  per_process_gpu_memory_fraction=0.3)))
loader = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'SSAE'))
loader.restore(sess, '/home/data0/dufan/Reconstruction/recon_new/train/KSAE/16x16-xy/k-%d-wd-0.1-f-1024-1024-1024/49'%sparsity_src)



# In[ ]:

res = SAEReconSQS(sino, img0, prj, ae, sess, hyper=50, nIter=200, strides=[8,8],
                  subStepSize=0.05, nSubSteps=5, random_patch=True, showSAELoss=True)


# In[ ]:



