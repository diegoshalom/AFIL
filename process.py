import os
import numpy as np
import SimpleITK as sitk
from scipy.ndimage import label
import pandas as pd

def calculate_volume_labels(labels, maxlabel):
    """    
    Calculate volumes of lesion labels

    Parameters
    ----------
    labels : np.array 
    maxlabel : maximum label number

    Returns
    -------
    y : np array
        volume
    xy : np array
        index

    """
    values = range(1, maxlabel + 2)
    y, x = np.histogram(labels.flatten(), values)
    x = np.delete(x, -1)

    return y, x

def small_lesion_filter(array):
    # Detect connected lesions
    labels, numlabels = label(array)
    
    # Calculate volumes
    vol1,labelid = calculate_volume_labels(labels, numlabels)
    
    # Remove small lesions from array
    indices_to_remove = np.in1d(labels, labelid[vol1<=3])
    indices_to_remove = np.reshape(indices_to_remove, np.shape(array))
    array[indices_to_remove] = False
    
    return array
    
datadir = './data/'
filename1 = 'P007-0m-Lesion_T1_BL_EDITED2.nii.gz'
filename2 = 'P007-3m-Lesion_T1_BL_EDITED2.nii.gz'

# Import lesion masks
img1 = sitk.ReadImage(os.path.join(datadir,filename1))
img2 = sitk.ReadImage(os.path.join(datadir,filename2))

# Transform to np.array
array1 = sitk.GetArrayFromImage(img1) > 0.5
array2 = sitk.GetArrayFromImage(img2) > 0.5

# Filter small lesions
array1 = small_lesion_filter(array1)
array2 = small_lesion_filter(array2)

# Stack images in 4D
matrix4d = np.stack((array1, array2))

# Detect connected lesions
labels, numlabels = label(matrix4d)

# Split labels 
labels1 = labels[0, :, :, :]
labels2 = labels[1, :, :, :]    

# Calculate volumes
vol1,labelid = calculate_volume_labels(labels1, numlabels)
vol2,labelid = calculate_volume_labels(labels2, numlabels)

# Export results as xlsx
data = np.transpose(np.vstack([labelid,vol1,vol2]))
data  = pd.DataFrame(data, columns=['id', 'v1', 'v2'])
xlsfname = os.path.join('volumes.xlsx')
data.to_excel(xlsfname, index=False)


