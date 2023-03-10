import scipy.ndimage as ndimage
import skimage.measure
import numpy as np

from torch.utils.data import Dataset

import os
import sys
import SimpleITK as sitk
import pydicom as pyd
import logging
from tqdm import tqdm

# import fill_voids
# import skimage.morphology

def write2pkl(file, dic) :
    from pickle import dump
    f=open( file + '.pkl', 'wb' )
    dump(dic,f)
    f.close() 
    
def readpkl( file ):
    from pickle import load
    file=open(file,'rb')
    dat = load(file )
    return dat

