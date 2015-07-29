from Preprocess import Image_Analyser as Imag,CIFAR as Cif, Kaggle as Kag
import numpy as np
import time




t1= time.time()
ahem = Cif.CIFAR()

tt = ahem.GeneratePatches()
print time.time() - t1

cent = ahem.computeKMeans()

cent = ahem.whittenMeans()