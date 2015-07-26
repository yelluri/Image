'''
Created on Jul 26, 2015

@author: vignesh
'''
import numpy as np
from random import randint
import logging
import scipy.misc
from Globals import *
from math import floor
class Image_Analyser(object):


    def __init__(self):
        '''
        Input the Image_size, Patch_size, Classes as an integer
        and Dataset_path as string for path
        This class is used to all computation on the image
        '''
        
        self.Image_size = IMAGE_SIZE
        self.Patch_size = PATCH_SIZE
        self.patch_set = np.empty((0,3*self.Patch_size*self.Patch_size),int)
        
    def readImage(self, filename):
        Image = scipy.misc.imread(filename)
        Image = self.resizeImage(Image)
        return Image
    
    def resizeImage(self, Image):
        newDim = np.array([self.Image_size,self.Image_size,3])
        resizedImage = scipy.misc.imresize(Image,newDim)
        return resizedImage
    
    def chooseRandomImage(self):
        label = np.array(['cat','dog'])
        fileName = "/home/vignesh/workspace/Kmeans/Datasets/Cats_Dogs/train/"+str(label[randint(0,1)])+"."+str(randint(0,10000))+".jpg"
        Image = self.readImage(fileName)
        return Image
        
    def computePatchAtRandom(self, Image):
        col_wise_total = floor(self.Image_size/self.Patch_size)
        row_wise_total = floor(self.Image_size/self.Patch_size)
        total_possible_patches = int(col_wise_total * row_wise_total)
        rand_patch_id = randint(0,total_possible_patches-1)#-1 since we assume from 0
        row = int(self.Patch_size * floor(rand_patch_id/col_wise_total))
        col = int(self.Patch_size * (rand_patch_id % col_wise_total))
        patch = Image[row:(row+self.Patch_size),col:(col+self.Patch_size),:].flatten()#Check this, may change based on the shape of given array. reshape if needed as reshape(PATCH_SIZE,PATCH_SIZE,3)
        return patch