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
from sklearn import cluster
from Preprocess import Image_Analyser as IA

from scipy import linalg




class CIFAR(IA.Image_Analyser):
    def __init__(self):
        super(CIFAR, self).__init__()
        
        self.Image_size = 32                                                                #CIFAR Images has Images of size 32X32 only
        self.Patch_size = PATCH_SIZE
        self.num_of_rand_patches = NO_OF_RANDOM_PATCHES
        self.KMeans = K_MEANS
        self.patch_set = np.empty((0,3*self.Patch_size*self.Patch_size),int)                #To collect the patches extracted
        self.col_wise_total = floor(self.Image_size/self.Patch_size)                        #No. of patches possible to extract in a column
        self.row_wise_total = floor(self.Image_size/self.Patch_size)                        #No. of patches possible to extract in a row
        self.total_possible_patches = int(self.col_wise_total * self.row_wise_total)        #Total No. of patches that can be extracted from the image
        self.Kmeans_iterations = K_MEANS_ITERATIONS
        
    
    def _chooseRandomImage(self):
        """Chooses an Image at random from the dataset"""
        
        Image = self.input_dataset[randint(0,49999)][:].reshape(3,self.Image_size,self.Image_size)  #Dataset has image matrix flatten, so we need to reshape
        return Image
        
        
    def _computePatchAtRandom(self, Image):
        """Based on the Image and patch size, the function extracts a patch and flattens it before returning"""
        
        rand_patch_id = randint(0,self.total_possible_patches-1)                            #-1 since we assume from 0-n
        row = int(self.Patch_size * floor(rand_patch_id/self.col_wise_total))               #find the first row id of the patch
        col = int(self.Patch_size * (rand_patch_id % self.col_wise_total))                  #find the first column id of the patch
        patch = Image[:,row:(row+self.Patch_size),col:(col+self.Patch_size)].flatten()      #Check this, may change based on the shape of given array. reshape if needed as reshape(3,PATCH_SIZE,PATCH_SIZE)
        
        return patch
    
    
    def _readDataset(self):
        """"Reads the Images which are stored as vectors in the binary file"""
        
        for i in xrange(5):                                                                 #There are 5 binary files to fetch the data
            fileObject = open("/home/vignesh/workspace/Kmeans/Datasets/CIFAR 10/cifar-10-batches-py/data_batch_"+str(i+1), 'rb')
            images = cPickle.load(fileObject)
            fileObject.close()
            if (i == 0):
                inputs_x = np.array(images['data'])                                         #Converting Dictionary to Numpy array for ease in manipulation.
                labels_y = np.array(images['labels'])
                imagenames = np.array(images['filenames'])
            else:
                inputs_x = np.vstack((inputs_x,np.array(images['data'])))                   
                labels_y = np.concatenate((labels_y,np.array(images['labels'])))
                imagenames = np.concatenate((imagenames,np.array(images['filenames'])))
                
        return inputs_x
    
    
    def GeneratePatches(self):
        """Generates a set of Random patches from the given set of images"""
        
        self.input_dataset = self._readDataset()                                            #Reads the data from the binary files
        for i in xrange(self.num_of_rand_patches):
            Image = self._chooseRandomImage()                                               #Chooses a random image from the dataset
            patch = self._computePatchAtRandom(Image)                                       #Chooses a patch from the Image at random
            self.patch_set = np.vstack((self.patch_set,patch))                              #collect these random patches in self.patch_set for clustering
            
        return self.patch_set
    
    
    def computeKMeans(self):
        """Computes K means Clustering and returns the K centroids"""
        
        #centroids, labels = scipy.cluster.vq.kmeans(self.patch_set,self.KMeans,self.Kmeans_iterations) #Compute Kmeans on the set of patches
        centroids = cluster.k_means(self.patch_set,self.KMeans)
        centroids = centroids[0]
        for i in xrange(centroids.shape[0]):                                                           #Save these means as images to visualize (also enlarge for better visualiztion)
            scipy.misc.imsave('CIFAR/Kmean/name'+str(i)+'.jpg', scipy.misc.imresize(centroids[i].reshape(3,PATCH_SIZE,PATCH_SIZE),(50,50)))
            
        return None
    
    
    def whittenMeans(self):
        """Whittens the data and Computes K means Clustering and returns the K centroids"""
        
        #pca_Obj = PCA(whiten=True)                                                              #Initialise the skLearn PCA decomposer for whittening the data           
        #transformed_patch_set = pca_Obj.fit_transform(self.patch_set.T)
        #transformed_patch_set = pca_Obj.inverse_transform(transformed_patch_set)
        transformed_patch_set = self.WhittenByPCA()
        #centroids, labels = scipy.cluster.vq.kmeans(np.real(transformed_patch_set.T),self.KMeans,self.Kmeans_iterations)   #K-means on this Whittened data
        centroids = cluster.k_means(np.real(transformed_patch_set.T),self.KMeans)
        centroids = centroids[0]
        for i in xrange(centroids.shape[0]):
            scipy.misc.imsave('CIFAR/Whitten/name'+str(i)+'.jpg', scipy.misc.imresize(centroids[i].reshape(3,PATCH_SIZE,PATCH_SIZE),(50,50)))
        print 'Computed and Saved the means'
        
        return None
    
    def WhittenByPCA(self):
        """Using PCA"""
        X = self.patch_set - np.mean(self.patch_set,axis=0)                                  
        X = X.T                                   
        Xcov=np.dot(X,X.T)                                          #Covariance Matrix
        print 'covar'
        val,vec=np.linalg.eigh(Xcov)                                #PCA decomposition
        print 'decompose'
        X  = np.dot(vec.T , X)                
        L = np.linalg.inv(linalg.sqrtm(np.diag(val+0.000001))) 
        print 'invert'
        X = np.dot(L,X)
        X = np.dot(vec,X)
        return X
    
    def Whitte(self):
        """Using SVD"""
        X = self.patch_set - np.mean(self.patch_set,axis=0)                             
        X = X.T
        print X.shape
        U, S, V = linalg.svd(X,full_matrices=False)
        print U.shape,V.shape
        X = np.dot(U.T,X)
        L = np.linalg.inv(np.diag(S))
        X = np.dot(L,X)
        #X = np.dot(U,X)
        
        return X