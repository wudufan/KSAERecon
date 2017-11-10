
# coding: utf-8

# In[ ]:

from ctypes import *
import os
import numpy as np
import numpy.fft
import numpy.matlib
import math
import matplotlib.pyplot as plt
import time
import re


# In[ ]:

class Projector:
    libprojectorEA = cdll.LoadLibrary(os.path.join(os.path.dirname(__file__), 'EAProjector_CPU.so'))
    libprojectionEACuda = cdll.LoadLibrary(os.path.join(os.path.dirname(__file__), 'libProjector.so'))
    
    def SetDevice(self, device):
        self.libprojectionEACuda.SetDevice(device);
    
    def __init__(self):
        self.rotview = 2304
        self.nu = 736
        self.nv = 103
        self.nx = 512
        self.ny = 512
        self.nz = 103
        self.dx = 1
        self.dy = 1
        self.dz = 1
        self.dsd = 1085.6
        self.dso = 595
        self.du = 1.2858
        self.dv = 1
        self.da = self.du/self.dsd
        self.off_a = (self.nu-1.0)/2 - 368.625
        self.off_u = 0
        self.off_v = 0        
        self.filtername = 'RL'
        self.cuda = True;
    
    def BackProjectionEA(self, sino, useIter=True):
        sino = sino.astype(np.float32)
        self.rotview = sino.shape[0]
        self.nu = sino.shape[1]
        img = np.zeros([self.nx,self.ny], dtype=np.float32)
        degs = np.asarray(range(0,self.rotview)).astype(np.float32) * 360/self.rotview
        
        if self.cuda is False:
            bpFunc = self.libprojectorEA.backprojectionkernel
        elif useIter:
            bpFunc = self.libprojectionEACuda.bpIterFan
        else:
            bpFunc = self.libprojectionEACuda.bpFan
            
        bpFunc(img.ctypes.data_as(POINTER(c_float)), sino.ctypes.data_as(POINTER(c_float)), 
               degs.ctypes.data_as(POINTER(c_float)), c_int(self.rotview), c_float(self.dsd), c_float(self.dso), 
               c_int(self.nx), c_int(self.ny), c_float(self.dx), c_float(self.dy), 
               c_int(self.nu), c_float(self.da), c_float(self.off_a))
        return img
    
    def BackProjectionParallel(self, sino):
        sino = sino.astype(np.float32)
        self.rotview = sino.shape[0]
        self.nu = sino.shape[1]
        img = np.zeros([self.nx,self.ny], dtype=np.float32)
        degs = np.asarray(range(0,self.rotview)).astype(np.float32) * 180/self.rotview
        
        bpFunc = self.libprojectionEACuda.bpParallel
            
        bpFunc(img.ctypes.data_as(POINTER(c_float)), sino.ctypes.data_as(POINTER(c_float)), 
               degs.ctypes.data_as(POINTER(c_float)), c_int(self.rotview), 
               c_int(self.nx), c_int(self.ny), c_float(self.dx), c_float(self.dy), 
               c_int(self.nu), c_float(self.du), c_float(self.off_u))
        return img
    
    def BPEA3D(self, sino, useIter=True):
        sino = sino.astype(np.float32)
        if self.nv != sino.shape[0] or self.rotview != sino.shape[1] or self.nu != sino.shape[2]:
            raise ValueError("sino shape (%d, %d, %d) does not match config shape (%d, %d, %d)"%                 (sino.shape[0], sino.shape[1], sino.shape[2], self.nv, self.rotview, self.nu))
        img = np.zeros([self.nz, self.ny, self.nx], dtype=np.float32)
        degs = np.asarray(range(0,self.rotview)).astype(np.float32) * 360/self.rotview
        
        if useIter:
            bpFunc = self.libprojectionEACuda.bpFanIter3D
        else:
            bpFunc = self.libprojectionEACuda.bpFan3D
        
        bpFunc(img.ctypes.data_as(POINTER(c_float)), sino.ctypes.data_as(POINTER(c_float)), 
               degs.ctypes.data_as(POINTER(c_float)), c_int(self.rotview), 
               c_float(self.dsd), c_float(self.dso), c_int(self.nx), c_int(self.ny), c_int(self.nz), 
               c_float(self.dx), c_float(self.dy), c_float(self.dz), 
               c_int(self.nu), c_int(self.nv), c_float(self.da), c_float(self.dv), 
               c_float(self.off_a), c_float(self.off_v))
        return img
        
    def ProjectionEA(self, img):
        img = img.astype(np.float32)
        sino = np.zeros([self.rotview, self.nu], dtype=np.float32)
        self.nx = img.shape[0]
        self.ny = img.shape[1]
        degs = np.asarray(range(0,self.rotview)).astype(np.float32) * 360/self.rotview
        
        if self.cuda is False:
            projFunc = self.libprojectorEA.projectionkernel
        else:
            projFunc = self.libprojectionEACuda.prjFan
        
        projFunc(sino.ctypes.data_as(POINTER(c_float)), img.ctypes.data_as(POINTER(c_float)), 
                 degs.ctypes.data_as(POINTER(c_float)), c_int(self.rotview), c_float(self.dsd), c_float(self.dso), 
                 c_int(self.nx), c_int(self.ny), c_float(self.dx), c_float(self.dy), 
                 c_int(self.nu), c_float(self.da), c_float(self.off_a))
        return sino
    
    def FPParallel(self, img):
        img = img.astype(np.float32)
        sino = np.zeros([self.rotview, self.nu], dtype=np.float32)
        self.nx = img.shape[0]
        self.ny = img.shape[1]
        degs = np.asarray(range(0,self.rotview)).astype(np.float32) * 180/self.rotview
        
        projFunc = self.libprojectionEACuda.fpParallel
        
        projFunc(sino.ctypes.data_as(POINTER(c_float)), img.ctypes.data_as(POINTER(c_float)), 
                 degs.ctypes.data_as(POINTER(c_float)), c_int(self.rotview), 
                 c_int(self.nx), c_int(self.ny), c_float(self.dx), c_float(self.dy), 
                 c_int(self.nu), c_float(self.du), c_float(self.off_u))
        return sino
    
    def FPEA3D(self, img):
        img = img.astype(np.float32)
        if self.nz != img.shape[0] or self.ny != img.shape[1] or self.nx != img.shape[2]:
            raise ValueError("image shape (%d, %d, %d) does not match config shape (%d, %d, %d)"%                             (img.shape[0], img.shape[1], img.shape[2], self.nz, self.ny, self.nx))
        
        sino = np.zeros([self.nv, self.rotview, self.nu], dtype = np.float32)
        degs = np.asarray(range(0,self.rotview)).astype(np.float32) * 360/self.rotview
        
        projFunc = self.libprojectionEACuda.fpFan3D
        projFunc(sino.ctypes.data_as(POINTER(c_float)), img.ctypes.data_as(POINTER(c_float)), 
                degs.ctypes.data_as(POINTER(c_float)), c_int(self.rotview), 
                c_float(self.dsd), c_float(self.dso), c_int(self.nx), c_int(self.ny), c_int(self.nz), 
                c_float(self.dx), c_float(self.dy), c_float(self.dz), 
                c_int(self.nu), c_int(self.nv), c_float(self.da), c_float(self.dv), 
                c_float(self.off_a), c_float(self.off_v))
        return sino
    
    def FilterPrjEAGPU(self, sino):
        sino= sino.astype(np.float32)
        if self.nv != sino.shape[0] or self.rotview != sino.shape[1] or self.nu != sino.shape[2]:
            raise ValueError("sino shape (%d, %d, %d) does not match config shape (%d, %d, %d)"%                             (sino.shape[0], sino.shape[1], sino.shape[2], self.nv, self.rotview, self.nu))
        
        fsino = np.zeros([self.nv, self.rotview, self.nu], dtype=np.float32)
        angles = (-np.arange(-self.nu / 2.0 + 0.5, self.nu / 2.0 + 0.5, 1) - self.off_a) * self.da
        angles = np.asarray(angles)
        
        if self.filtername.lower() == "hamming":
            print 'hamming'
            filtertype = 1
        elif self.filtername.lower() == 'hann':
            print 'hann'
            filtertype = 2
        elif self.filtername.lower() == 'cosine':
            print 'cosine'
            filtertype = 3
        else:
            filtertype = 0
        
        filterFunc = self.libprojectionEACuda.FilterEA
        filterFunc(fsino.ctypes.data_as(POINTER(c_float)), sino.ctypes.data_as(POINTER(c_float)), 
                c_int(self.rotview), c_float(self.dsd), c_float(self.dso), 
                c_int(self.nu), c_int(self.nv), c_float(self.da), c_float(self.off_a), filtertype)
        return fsino
    
    def FilterPrjParallelGPU(self, sino):
        sino= sino.astype(np.float32)
        if self.nv != sino.shape[0] or self.rotview != sino.shape[1] or self.nu != sino.shape[2]:
            raise ValueError("sino shape (%d, %d, %d) does not match config shape (%d, %d, %d)"%                             (sino.shape[0], sino.shape[1], sino.shape[2], self.nv, self.rotview, self.nu))
        
        fsino = np.zeros([self.nv, self.rotview, self.nu], dtype=np.float32)
        
        if self.filtername.lower() == "hamming":
            filtertype = 1
            print 'hamming'
        else:
            filtertype = 0
        
        filterFunc = self.libprojectionEACuda.FilterParallel
        filterFunc(fsino.ctypes.data_as(POINTER(c_float)), sino.ctypes.data_as(POINTER(c_float)), 
                c_int(self.rotview), 
                c_int(self.nu), c_int(self.nv), c_float(self.du), c_float(self.off_u), filtertype)
        return fsino
    
    def FilterProjectionEA(self, sino):
        self.rotview = sino.shape[0]
        self.nu = sino.shape[1]
        
        angles = (-np.arange(-self.nu / 2.0 + 0.5, self.nu / 2.0 + 0.5, 1) - self.off_a) * self.da
        angles = np.asarray(angles)
        
        for i in range(0,self.rotview):
            sino[i, :] *= np.cos(angles/180*math.pi)
        
        filt_len = 2 * self.nu - 1;
        h = self.rampEA(filt_len)
        fh = np.fft.rfft(h)
        fsino = np.fft.rfft(np.concatenate((sino, np.zeros([sino.shape[0], self.nu-1])), axis=-1))
        fsino = fsino * np.matlib.repmat(fh, self.rotview, 1)
        sino = np.real(np.fft.irfft(fsino, n = filt_len))
        
        # (pi/rotview) is for integral step
        # self.da is for convolution step
        # /self.dso is because the CBFactor in the bp kernel has an additional dso on the numerator
        sino = sino[:, self.nu-1:] * (math.pi / self.rotview) * self.da / self.dso
        
        return sino.astype(np.float32)
    
    def rampEA(self, n):
        nn = np.arange(-(int(n) / 2), int(n) / 2 + 1, dtype=int)
        h = np.zeros(len(nn), dtype=np.float32)
        h[0] = 1
        h[int(n) / 2] = 1 / (4 * self.da * self.da);
        odd = (nn%2 == 1)
        h[odd] = -1 / (math.pi * np.sin(nn[odd] * self.da))**2
        
        print self.filtername.lower()
        if self.filtername.lower() is 'hamming':
            print 'Hamming window'
            w = 0.54 + 0.46 * np.cos(2*pi*np.arange(0,len(h)) / (len(h) - 1))
        else:
            w = np.ones(h.shape)
        
        h = np.real(np.fft.ifft(np.fft.fft(h) * w))
        
        return h
    
    def FromFile(self, filename):
        f = open(filename, 'r')
        params = dict()
        for line in f:
            substrs = re.split(' |\t|\n', line)
            name = substrs[0].lower()
            var = np.float32(substrs[1])
            params[name] = var
        for varname in dir(self):
            if varname in params:
                setattr(self, varname, params[varname])
        self.rotview = int(self.rotview)
        self.nu = int(self.nu)
        self.nx = int(self.nx)
        self.ny = int(self.ny)
        self.off_a = -self.off_a
        


# In[ ]:

class TotalVariation:
    eps = 1e-6
    
    @staticmethod
    def TVGrad2D(img):
        padimg = np.zeros([img.shape[0]+2, img.shape[1]+2])
        padimg[1:-1, 1:-1] = img
        dx = padimg[1:-1, 1:-1] - padimg[:-2, 1:-1]
        dy = padimg[1:-1 ,1:-1] - padimg[1:-1, 2:]
        dxx = padimg[2:, 1:-1] - padimg[1:-1, 1:-1]
        dyx = padimg[2:, 1:-1] - padimg[2:, :-2]
        dxy = padimg[1:-1, 2:] - padimg[:-2, 2:]
        dyy = padimg[1:-1, 2:] - padimg[1:-1, 1:-1]
        
        tvgrad = (dx + dy) / np.sqrt(dx*dx + dy*dy + TotalVariation.eps)                 -dxx / np.sqrt(dxx*dxx + dyx*dyx + TotalVariation.eps)                 -dyy / np.sqrt(dxy*dxy + dyy*dyy + TotalVariation.eps)
        
        
        return tvgrad
    
    @staticmethod
    def Variation2D(img):
        padimg = np.zeros([img.shape[0]+2, img.shape[1]+2], np.float32)
        padimg[1:-1, 1:-1] = img
        dx = padimg[1:-1, 1:-1] - padimg[:-2, 1:-1]
        dy = padimg[1:-1, 1:-1] - padimg[1:-1, :-2]
        
        return dx, dy
    
    @staticmethod
    def TotalVariation2D(img):
        return np.sum(np.abs(dx)) + np.sum(np.abs(dy))
    
    @staticmethod
    def Variation2DTranspose((imgx, imgy)):
        padimgx = np.zeros([imgx.shape[0]+2, imgx.shape[1]+2], np.float32)
        padimgy = np.zeros([imgy.shape[0]+2, imgy.shape[1]+2], np.float32)
        padimgx[1:-1, 1:-1] = imgx
        padimgy[1:-1, 1:-1] = imgy
        dx = padimgx[1:-1, 1:-1] - padimgx[2:, 1:-1]
        dy = padimgy[1:-1, 1:-1] - padimgy[1:-1, 2:]
        
        return dx + dy
    
    @staticmethod
    def L2Grad2D(img):
        padimg = np.zeros([img.shape[0]+2, img.shape[1]+2])
        padimg[1:-1, 1:-1] = img
        dx = padimg[1:-1, 1:-1] - padimg[:-2, 1:-1]
        dy = padimg[1:-1 ,1:-1] - padimg[1:-1, 2:]
        dxx = padimg[2:, 1:-1] - padimg[1:-1, 1:-1]
        dyy = padimg[1:-1, 2:] - padimg[1:-1, 1:-1]
        
        tvgrad = dx + dy - dxx - dyy
        
        
        return tvgrad


# In[ ]:

class fan3D:
    lib = cdll.LoadLibrary(os.path.join(os.path.dirname(__file__), 'libProjector.so'))
    
    def SetWorker(self, worker):
        return self.lib.fan3DSelectWorker(c_int(worker))
    
    def TouchWorker(self):
        return self.lib.fan3DTouchWorker()
        
    def DestroyWorker(self):
        return self.lib.fan3DDestroyWorker()
        
    def DestroyAllWorkers(self):
        return self.lib.fan3DDestroyAllWorker()
    
    def __init__(self):
        self.rotview = 2304
        self.nu = 736
        self.nv = 103
        self.nx = 512
        self.ny = 512
        self.nz = 103
        self.dx = 1
        self.dy = 1
        self.dz = 1
        self.dsd = 1085.6
        self.dso = 595
        self.du = 1.2858
        self.dv = 1
        self.da = self.du/self.dsd
        self.off_a = (self.nu-1.0)/2 - 368.625
        self.off_v = 0
        self.lib.fan3DSQSData.restype = c_float
        self.lib.fan3DRetrieveTVSQS.restype = c_float
        
        
    def FromFile(self, filename):
        f = open(filename, 'r')
        params = dict()
        for line in f:
            substrs = re.split(' |\t|\n', line)
            name = substrs[0].lower()
            var = np.float32(substrs[1])
            params[name] = var
        for varname in dir(self):
            if varname in params:
                setattr(self, varname, params[varname])
        self.rotview = int(self.rotview)
        self.nu = int(self.nu)
        self.nv = int(self.nv)
        self.nx = int(self.nx)
        self.ny = int(self.ny)
        self.nz = int(self.nz)
        self.off_a = -self.off_a
        
    def Setup(self, device):
        degs = np.arange(0,self.rotview, dtype=np.float32) * 360/self.rotview
        return self.lib.fan3DSetup(c_int(device), degs.ctypes.data_as(POINTER(c_float)), 
                       c_int(self.nx), c_int(self.ny), c_int(self.nz), 
                       c_float(self.dx), c_float(self.dy), c_float(self.dz),
                       c_int(self.nu), c_int(self.nv), c_int(self.rotview),
                       c_float(self.dsd), c_float(self.dso), 
                       c_float(self.da), c_float(self.dv), c_float(self.off_a), c_float(self.off_v))
    
    def SyncPrjFromCPU(self, sino):
        return self.lib.fan3DSyncPrjFromCPU(sino.ctypes.data_as(POINTER(c_float)))
    
    def SyncPrjToCPU(self):
        sino = np.zeros([self.nv, self.rotview, self.nu], dtype=np.float32)
        self.lib.fan3DSyncPrjToCPU(sino.ctypes.data_as(POINTER(c_float)))
        return sino
    
    def SyncImgFromCPU(self, img):
        return self.lib.fan3DSyncImgFromCPU(img.ctypes.data_as(POINTER(c_float)))
    
    def SyncImgToCPU(self):
        img = np.zeros([self.nz, self.ny, self.nx], dtype=np.float32)
        self.lib.fan3DSyncImgToCPU(img.ctypes.data_as(POINTER(c_float)))
        return img
    
    def LoadZerosToPrj(self):
        self.lib.fan3DLoadZerosToPrj()
    
    def LoadZerosToImg(self):
        self.lib.fan3DLoadZerosToImg()
    
    def ForwardProjection(self):
        self.lib.fan3DForwardProjection()
    
    def BackProjection(self):
        self.lib.fan3DBackProjection()
    
    def FBP(self, filtertype=0):
        self.lib.fan3DFBP(c_int(filtertype))
    
    def GetMatrixNorm(self, nIter=20, outputOn = False):
        if outputOn is True:
            valOutputOn = 1
        else:
            valOutputOn = -1
        self.lib.fan3DGetMatrixNorm.restype = c_float
        norm = self.lib.fan3DGetMatrixNorm(c_int(nIter), c_int(valOutputOn))
        return norm
    
    def SARTInit(self):
        return self.lib.fan3DSARTInit()
    
    def SART(self, lam=1):
        self.lib.fan3DSART.restype = c_float
        err = self.lib.fan3DSART(c_float(lam))
        return err
        
    def SARTDestroy(self):
        self.lib.fan3DSARTDestroy()
    
    def POCSPositive(self):
        self.lib.fan3DPOCSPositive()
    
    def Landweber(self, norm, lam=1):
        self.lib.fan3DLandweber.restype = c_float
        err = self.lib.fan3DLandweber(c_float(norm), c_float(lam))
        return err
    
    def TV(self, step):
        return self.lib.fan3DTV(c_float(step))
    
    def L2(self, step):
        return self.lib.fan3DL2(c_float(step))
    
    def RetrieveGradientLandweber(self, normalized=True):
        if normalized is True:
            valNorm = 1
        else:
            valNorm = -1;
        grad = np.zeros([self.nz, self.ny, self.nx], dtype=np.float32)
        self.lib.fan3DRetrieveGradientLanweber(grad.ctypes.data_as(POINTER(c_float)), c_int(valNorm))
        return grad
    
    def RetrieveGradientTV(self, normalized=True):
        if normalized is True:
            valNorm = 1
        else:
            valNorm = -1;
        grad = np.zeros([self.nz, self.ny, self.nx], dtype=np.float32)
        self.lib.fan3DRetrieveGradientTV(grad.ctypes.data_as(POINTER(c_float)), c_int(valNorm))
        return grad
    
    def RetrieveGradientL2(self, normalized=True):
        if normalized is True:
            valNorm = 1
        else:
            valNorm = -1;
        grad = np.zeros([self.nz, self.ny, self.nx], dtype=np.float32)
        self.lib.fan3DRetrieveGradientL2(grad.ctypes.data_as(POINTER(c_float)), c_int(valNorm))
        return grad
    
    def GetProjectionErr(self):
        self.lib.fan3DGetProjectionErr.restype = c_float
        return self.lib.fan3DGetProjectionErr()
    
    def GetTVCostFunc(self):
        self.lib.fan3DGetTVCostFunc.restype = c_float
        return self.lib.fan3DGetTVCostFunc()
    
    def GetL2CostFunc(self):
        self.lib.fan3DGetL2CostFunc.restype = c_float
        return self.lib.fan3DGetL2CostFunc()
    
    def RetrieveTVSQS(self):
        sumDiff = np.zeros([self.nz, self.ny, self.nx], np.float32)
        sumNorm = np.zeros([self.nz, self.ny, self.nx], np.float32)
        cf = self.lib.fan3DRetrieveTVSQS(sumDiff.ctypes.data_as(POINTER(c_float)), 
                                         sumNorm.ctypes.data_as(POINTER(c_float)))
        return sumDiff, sumNorm, cf
    
    def SQSGetWeights(self, sino):
        weights = np.zeros(sino.shape, np.float32)
        dataNorm = np.zeros([self.nz, self.ny, self.nx], np.float32)
        err = self.lib.fan3DSQSGetWeights(weights.ctypes.data_as(POINTER(c_float)), 
                                          dataNorm.ctypes.data_as(POINTER(c_float)), 
                                          sino.ctypes.data_as(POINTER(c_float)))
        
        return weights, dataNorm, err
    
    def SQSInit(self, weights):
        return self.lib.fan3DSQSInit(weights.ctypes.data_as(POINTER(c_float)))
    
    def SQSDestroy(self):
        self.lib.fan3DSQSDestroy()
        
    def SQSData(self):
        sumDiff = np.zeros([self.nz, self.ny, self.nx], np.float32)
        cf = self.lib.fan3DSQSData(sumDiff.ctypes.data_as(POINTER(c_float)))
        
        return sumDiff, cf


# In[ ]:



