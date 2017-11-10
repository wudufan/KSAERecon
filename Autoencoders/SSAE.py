
# coding: utf-8

# In[1]:

import tensorflow as tf
import numpy as np


# In[4]:

# stacked sparse autoencoder

class StackedSparseAutoEncoder:
    # imageshape: shape of patch, in x,y,z
    # nFeatures: # of features for each level of encoder, number of decoders are the same
    # sparsity: sparsity parameter for different number of stacks. This is useful when the encoder is built in 
    #    a stacked way. If built in a finetune way, only the last in the list is used.
    # weight_decay: weight decay
    # mode: 1 for L1 sparse, then the sparsity parameter means the penalty weights; 
    #       0 for K sparse, then the sparsity parameter means the number of non-zero elements for each level of encoder
    def __init__(self, imgshape=[16,16,1], nFeatures=[1024,1024,1024], sparsity=[1,10,100], weight_decay=0.1, mode=1):
        self.imgshape = imgshape
        self.imgsize = imgshape[0] * imgshape[1]* imgshape[2]
        self.nFeatures = nFeatures
        self.sparsity = sparsity
        self.weight_decay = weight_decay
        self.mode = mode # 0 for K sparse, 1 for L1 sparse
    
    # build up encoder
    def Encoder(self, input_data, scope='encoder', reuse=False, nFeatures=None):
        with tf.variable_scope(scope, reuse = reuse):
            if nFeatures is None:
                nFeatures = self.nFeatures

            encode_datas = list()
            encode_datas.append(input_data)
            h = tf.contrib.layers.flatten(input_data)            
                        
            for i in range(len(nFeatures)):
                h = tf.layers.dense(h, nFeatures[i], tf.nn.relu, name='fc%d'%i)
                encode_datas.append(h)
        
        with tf.variable_scope(scope, reuse = True):
            encoder_weights = list()
            encoder_biases = list()
            for i in range(len(nFeatures)):
                encoder_weights.append(tf.get_variable('fc%d/kernel'%i))
                encoder_biases.append(tf.get_variable('fc%d/bias'%i))
        
        return encode_datas, encoder_weights, encoder_biases
    
    #build up decoder
    def Decoder(self, encode_data, scope='decoder', reuse=False, nFeatures=None):
        with tf.variable_scope(scope, reuse=reuse):
            if nFeatures is None:
                nFeatures = self.nFeatures[:-1]
                        
            decode_datas=list()
            h = encode_data
            decode_datas.append(h)
                    
            for i in range(len(nFeatures), 0, -1):
                h = tf.layers.dense(h, nFeatures[i-1], tf.nn.relu, name='fc%d'%i)
                decode_datas.append(h)
            h = tf.layers.dense(h, self.imgsize, name='fc0')
            decode_datas.append(tf.reshape(h, [tf.shape(h)[0]] + self.imgshape))
        
        with tf.variable_scope(scope, reuse = True):
            decoder_weights = list()
            decoder_biases = list()
            for i in range(len(nFeatures), -1, -1):
                decoder_weights.append(tf.get_variable('fc%d/kernel'%i)) 
                decoder_biases.append(tf.get_variable('fc%d/bias'%i)) 
        
        return decode_datas, decoder_weights, decoder_biases
    
    # build the stacked autoencoder
    # iStack: number of stacks to use (<= len(nFeatures)), this is useful when training layer by layer (which was 
    #    not used in the TMI paper)
    def BuildStackedAutoEncoder(self, iStack=-1, scope='SSAE', reuse=False):
        with tf.variable_scope(scope, reuse=reuse):
            if iStack < 0 or iStack >= len(self.nFeatures):
                iStack = len(self.nFeatures)-1
            nFeatures = self.nFeatures[:(iStack+1)]
            sparsity = self.sparsity[iStack]
            
            self.input_data = tf.placeholder(tf.float32, [None] + self.imgshape, name='input')
            
            self.encode_datas, encoder_weights, encoder_biases =                 self.Encoder(self.input_data, scope='encoder', reuse=reuse, nFeatures=nFeatures)
            
            # explicitly apply K sparse constrains on the uppermost encoded layer
            if self.mode == 0:
                # k-sparse
                self.encode_datas[-1] = self.KSparseMask(self.encode_datas[-1], sparsity)
            
            self.decode_datas, decoder_weights, decoder_biases =                 self.Decoder(self.encode_datas[-1], scope='decoder', reuse=reuse, nFeatures=nFeatures[:-1])
            
            # build stack-wise losses, the features recovered by a decoder was compared to the  
            #   features input to the corresponding encoder
            self.losses = list()
            for i in range(len(self.encode_datas)):
                loss = tf.sqrt(tf.reduce_mean((self.encode_datas[i] - self.decode_datas[len(self.decode_datas)-i-1])**2))
                self.losses.append(loss)
            
            self.loss_img = self.losses[0]  # image loss
            self.loss_upmost = self.losses[-2]  # the upmost feature loss, useful for stacked training
            self.loss_sparse = tf.reduce_mean(tf.abs(self.encode_datas[-1]))  # sparsity loss for L1 sparse
            
            # weight decay
            self.loss_weight = 0
            w_count = 0
            for w in encoder_weights:
                self.loss_weight += tf.reduce_mean(w**2)
                w_count += 1
            for w in decoder_weights:
                self.loss_weight += tf.reduce_mean(w**2)
                w_count += 1
            self.loss_weight = tf.sqrt(self.loss_weight / w_count)
            
            # total loss
            if self.mode == 0:
                self.loss_current = self.loss_upmost + self.weight_decay * self.loss_weight
                self.loss_total = self.loss_img + self.weight_decay * self.loss_weight
            else:
                self.loss_current = self.loss_upmost + sparsity * self.loss_sparse + self.weight_decay * self.loss_weight
                self.loss_total = self.loss_img + sparsity * self.loss_sparse + self.weight_decay * self.loss_weight
            
            # vars
            self.vars_encoder = encoder_weights + encoder_biases
            self.vars_decoder = decoder_weights + decoder_biases
            self.vars_upmost = [encoder_weights[-1], encoder_biases[-1], decoder_weights[0], decoder_biases[0]]
    
    # select the K largest element and set the rest to zero. it should be only computed during forward propagation
    def KSparseMask(self, encode_data, sparsity, scope='SSAE', reuse=False):
        with tf.variable_scope(scope, reuse):
            h = encode_data
            
            _, indices = tf.nn.top_k(tf.abs(h), k=sparsity, name='top_k')
            indices_dim1 = tf.expand_dims(tf.range(0, tf.shape(h)[0]), 1)
            indices_dim1 = tf.tile(indices_dim1, [1, tf.shape(indices)[-1]])
            full_indices = tf.concat([tf.expand_dims(indices_dim1, 2), tf.expand_dims(indices, 2)], 2)
            full_indices = tf.reshape(full_indices, [-1, 2])
            mask = tf.sparse_to_dense(full_indices, tf.shape(h), 1.0, validate_indices=False)
            h = tf.multiply(h, mask)
            
            return h
    
    # given l = (y-f(x))^2, calculate dl / dx
    def BuildGradientsWRTInput(self, scope='SSAE', reuse=False):
        with tf.variable_scope(scope, reuse):
            self.ref_data = tf.placeholder(tf.float32, [None] + self.imgshape, 'input_latent')
            self.loss_ref = tf.sqrt(tf.reduce_mean((self.ref_data - self.decode_datas[-1])**2))
            
            self.grad_ref = tf.gradients(self.loss_ref, self.input_data)[0]
            self.grad_sparse = tf.gradients(self.loss_sparse, self.input_data)[0]
            self.grad_loss = tf.gradients(self.loss_img, self.input_data)[0]
            
            if self.mode == 0:
                self.loss_ref_total = self.loss_ref
                self.grad_ref_total = self.grad_ref
            else:
                self.loss_ref_total = self.loss_ref + self.sparsity[-1] * self.loss_sparse
                self.grad_ref_total = self.grad_ref + self.sparsity[-1] * self.grad_sparse
    
    # predict f(x) for patches
    def Predict(self, patches, batchsize, sess):
        res_patches = np.zeros(patches.shape, np.float32)
        for i in range(0, patches.shape[0], batchsize):
            batch = patches[i:i+batchsize,...]
            [res] = sess.run([self.decode_datas[-1]], feed_dict = {self.input_data: batch[...,np.newaxis]})
            res_patches[i:i+batchsize,...] = res.squeeze()
        
        return res_patches
    
    # get the gradient (actual calculation)
    def GetRefGradients(self, patches, ref_patches, batchsize, sess):
        grads = np.zeros(patches.shape, np.float32)
        for i in range(0, patches.shape[0], batchsize):
            batch = patches[i:i+batchsize,...]
            ref_batch = ref_patches[i:i+batchsize,...]
            [grad] = sess.run([self.grad_ref_total], 
                              feed_dict = {self.input_data: batch[...,np.newaxis], 
                                           self.ref_data: ref_batch[...,np.newaxis]})
            grads[i:i+batchsize,...] = grad.squeeze()
        
        return grads
    
    # get the loss (y-f(x))^2 (actual calculation)
    def GetRefLoss(self, patches, refPatches, batchsize, sess):
        vals = list()
        for i in range(0, patches.shape[0], batchsize):
            batch = patches[i:i+batchsize,...]
            refBatch = refPatches[i:i+batchsize,...]
            [val] = sess.run([self.loss_ref_total], 
                             feed_dict = {self.input_data: batch[...,np.newaxis], 
                                          self.ref_data: refBatch[...,np.newaxis]})
            vals.append(val)
        return sum(vals) / len(vals)
    
    # grey scale range transform, for patch grey scale range normalization
    def MapGreyScaleRange(self, img, vmin, vmax, vmin_new, vmax_new, crop = True):
        a = (vmax * vmin_new - vmin * vmax_new) / (vmax_new - vmin_new)
        b = (vmax_new - vmin_new) / (vmax - vmin)
        res = (img + a) * b
        if crop is True:
            res[res < vmin_new] = vmin_new
            res[res > vmax_new] = vmax_new
        
        return res
        


# In[ ]:



