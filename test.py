from Preprocess import Image_Analyser as Imag,CIFAR as Cif, Kaggle as Kag
import numpy as np
import time
from scipy import linalg
import matplotlib.pyplot as plt

# 
# a1 = np.random.random_integers(10., size=(35.,2.))
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.set_ylim(-1,2)
# ax.set_xlim(-1,2)
# #ax.plot(a1[:,0],a1[:,1],'ro')
# b1 = a1.mean(axis=0) - a1
# print b1.mean(axis=0)
# ax.plot(b1[:,0],b1[:,1],'go')
# X=b1.T
# Xcov=np.dot(X,X.T)
# val,vec=np.linalg.eigh(Xcov)
# X  = np.dot(vec.T , X)
# X=X.T
# ax.plot(X[:,0],X[:,1],'bo')
# X=X.T
# L = np.linalg.inv(linalg.sqrtm(np.diag(val))) 
# X = np.dot(L,X)
# X=X.T
# ax.plot(X[:,0],X[:,1],'ro')
# X=X.T
# X = np.dot(vec,X)
# X=X.T
# ax.plot(X[:,0],X[:,1],'go')
# plt.show()
# assert 0,'D'


t1= time.time()
ahem = Kag.Kaggle()

tt = ahem.GeneratePatches()
print time.time() - t1

cent = ahem.computeKMeans()

cent = ahem.whittenMeans()