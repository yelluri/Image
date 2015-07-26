import scipy.misc
import numpy as np
import Image
from random import randint

Imd = scipy.misc.imread("/home/vignesh/workspace/Kmeans/Datasets/Cats_Dogs/train/cat.14.jpg")
print Imd.shape
scipy.misc.imsave('cat.jpg',Imd)
dim = np.array([128,128,3])
Im2 = scipy.misc.imresize(Imd,dim)
print Im2.shape
scipy.misc.imsave('cat2.jpg',Im2)
label = np.array(['Cat','Dog'])
print label[randint(0,1)]
fileName = "/home/vignesh/workspace/Kmeans/Datasets/Cats_Dogs/train/"+str(label[randint(0,1)])+"."+str(randint(0,10000))+".jpg"
print fileName
        