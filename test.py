from Preprocess import CIFAR_DataSet as CIFAR
import numpy as np
from scipy import misc
import Image
import random
from math import floor
size=(64,64,3)
Obj = CIFAR.CIFAR_DataSet()
x,y,z = Obj.get_DataSet()

ss = Obj.compute_patch(8,4,x[100][:])
#print ss.shape

temp = np.arange(48).reshape(3,4,4)
row = 0
col = 1
print temp
tt=temp[:,row:(row+2),col:(col+2)].flatten()
print tt.shape,tt,tt.flatten()

temp = np.arange(48).reshape(3,4,4)
t1 = (temp+3).flatten()
print temp
print t1
tt=temp[:,row:(row+2),col:(col+2)].flatten()
print tt,t1[tt]