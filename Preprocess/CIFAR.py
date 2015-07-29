'''
Created on Jul 26, 2015

@author: vignesh
'''
import numpy as np
from random import randint
import cPickle
import scipy.misc
from Globals import *
from math import floor
import scipy.cluster
from sklearn.decomposition import PCA
from Preprocess import Image_Analyser as IA

class CIFAR(IA.Image_Analyser):
    def __init__(self):
        super(CIFAR, self).__init__()
        
        self.Image_size = 32
        self.Patch_size = PATCH_SIZE
        self.num_of_rand_patches = NO_OF_RANDOM_PATCHES
        self.KMeans = K_MEANS
        self.patch_set = np.empty((0,3*self.Patch_size*self.Patch_size),int)
        self.col_wise_total = floor(self.Image_size/self.Patch_size)
        self.row_wise_total = floor(self.Image_size/self.Patch_size)
        self.total_possible_patches = int(self.col_wise_total * self.row_wise_total)
        
    
    def chooseRandomImage(self):
        Image = self.input_dataset[randint(0,9999)][:].reshape(3,self.Image_size,self.Image_size)
        return Image
        
    def computePatchAtRandom(self, Image):
        rand_patch_id = randint(0,self.total_possible_patches-1)#-1 since we assume from 0
        row = int(self.Patch_size * floor(rand_patch_id/self.col_wise_total))
        col = int(self.Patch_size * (rand_patch_id % self.col_wise_total))
        patch = Image[:,row:(row+self.Patch_size),col:(col+self.Patch_size)].flatten()#Check this, may change based on the shape of given array. reshape if needed as reshape(PATCH_SIZE,PATCH_SIZE,3)
        return patch
    
    
    def readDataset(self):
        fileObject = open("/home/vignesh/workspace/Kmeans/Datasets/CIFAR 10/cifar-10-batches-py/data_batch_1", 'rb')
        images = cPickle.load(fileObject)
        fileObject.close()
        #Converting Dictionary to Numpy array for ease in manipulation.
        inputs_x = np.array(images['data'])
        #labels_y = np.array(images['labels'])
        #imagenames = np.array(images['filenames'])  
        return inputs_x
    
    def GeneratePatches(self):
        self.input_dataset = self.readDataset()
        for i in xrange(self.num_of_rand_patches):
            Image = self.chooseRandomImage()
            patch = self.computePatchAtRandom(Image)
            self.patch_set = np.vstack((self.patch_set,patch))#input data set whose type is a list
        return self.patch_set
    
    def computeKMeans(self):
        centroids, labels = scipy.cluster.vq.kmeans(self.patch_set,self.KMeans,1)
        for i in xrange(centroids.shape[0]):
            scipy.misc.imsave('CIFAR/Kmean/name'+str(i)+'.jpg', scipy.misc.imresize(centroids[i].reshape(3,PATCH_SIZE,PATCH_SIZE),(50,50)))
        return None
    
    def whittenMeans(self):
        #data = scipy.cluster.vq.whiten(self.patch_set)
#         X=self.patch_set.T                                          #For PCA, data should be in dimXno. of samples
#         X = X-np.mean(X,axis=0)                                     #Data to Zero mean
#         Xcov=np.dot(X,X.T)                                          #Covariance Matrix
#         print 'covar'
#         val,vec=np.linalg.eig(Xcov)                                #PCA decomposition
#         print 'decompose'
#         Xcov  = np.dot(vec.T , X)                                   
#         eig = np.diag(val)                                          #Diagonalize the matrix
#         L = np.linalg.inv(scipy.linalg.sqrtm(eig))                  #find eigenval**-0.5
#         print 'invert'
#         Xcov = np.dot(L,Xcov)
#         Xcov = np.dot(vec,Xcov)
        Xcov = 0
        pca = PCA(whiten=True)
        transformed = pca.fit_transform(self.patch_set.T)
        centroids, labels = scipy.cluster.vq.kmeans(np.real(transformed.T),self.KMeans,1)   #K-means on this Whittened data
        for i in xrange(centroids.shape[0]):
            scipy.misc.imsave('CIFAR/Whitten/name'+str(i)+'.jpg', scipy.misc.imresize(centroids[i].reshape(3,PATCH_SIZE,PATCH_SIZE),(50,50)))
        print 'Computed and Saved the means'
        return None