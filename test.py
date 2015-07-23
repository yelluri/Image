from Preprocess import CIFAR_DataSet as CIFAR
import numpy as np
from scipy import misc
import Image
import random
from math import floor
size=(64,64,3)
Obj = CIFAR.CIFAR_DataSet()
x,y,z = Obj.get_DataSet()
print 
ss = Obj.compute_patch(8,2,x[100][:])
print ss.shape

