import cPickle
import random
import numpy as np
from math import floor

class CIFAR_DataSet(object):  
    def __init__(self):
        self.dataset_name = "CIFAR Dataset With 10 Classes"
    
    def get_DataSet(self):
        fileObject = open("Datasets/CIFAR 10/cifar-10-batches-py/data_batch_1", 'rb')
        images = cPickle.load(fileObject)
        fileObject.close()
        #Converting Dictionary to Numpy array for ease in manipulation.
        inputs_x = np.array(images['data'])
        labels_y = np.array(images['labels'])
        imagenames = np.array(images['filenames'])  
        return inputs_x, labels_y, imagenames
    
    def compute_patch(self,image_size , patch_size , x):
        """Computes the sub-image or patch from the given image"""
     
        randomNum = random.randint ( 0 , ( ( image_size ** 2 ) / ( patch_size ** 2 ) ) - 1 )
        row = patch_size * floor( ( randomNum * patch_size ) / image_size ) 
        col = ( randomNum * patch_size ) % image_size
        startIndex = ( row * image_size ) + col
        patch = self.getIndices(image_size,patch_size,int(startIndex)) 
        return x[patch]
     
    def getIndices(self,image_size,patch_size,startIndex):
        """"Computes the Indices from which the patch values can be taken"""
        
        for i in xrange(patch_size):
            tempArray = ( startIndex + ( i * image_size ) ) + np.arange( patch_size )
            if i==0:
                patch = tempArray
            else:
                patch = np.concatenate((  patch , tempArray ))
        temp = (image_size * image_size) + patch
        temp = np.concatenate((temp,((image_size*image_size)+temp)))
        patch = np.concatenate( ( patch , temp) )
        print patch
        return patch