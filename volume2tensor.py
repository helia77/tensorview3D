# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 15:49:53 2024

@author: helioum
"""
import sys
import numpy as np
from scipy import signal
from skimage.io import imread


def structure3d(T,  sigma, noise=0, SAVE=False):
    """
    This function takes a 3D grayscale volume and returns the structure tensor
    for visualization in tensorview3D.

    Parameters:
    T (uint8): the input grayscale volume (ZxYxX)

    Returns:
    float32: the structure tensor (ZxYxXx3x3)
    """
    # create the structure tensor
    dVdz, dVdy, dVdx = np.gradient(T, edge_order=2)
    T = np.zeros((dVdx.shape[0], dVdx.shape[1], dVdx.shape[2], 3, 3))
    T[:, :, :, 0, 0] = dVdx * dVdx
    if noise > 0:
        T[:, :, :, 0, 0] += np.abs(np.random.normal(0.0, noise, T[:, :, :, 0, 0].shape))
    T[:, :, :, 1, 1] = dVdy * dVdy
    if noise > 0:
        T[:, :, :, 1, 1] += np.abs(np.random.normal(0.0, noise, T[:, :, :, 1, 1].shape))
    T[:, :, :, 2, 2] = dVdz * dVdz
    if noise > 0:
        T[:, :, :, 2, 2] += np.abs(np.random.normal(0.0, noise, T[:, :, :, 2, 2].shape))
    T[:, :, :, 0, 1] = dVdx * dVdy
    if noise > 0:
        T[:, :, :, 0, 1] += np.abs(np.random.normal(0.0, noise, T[:, :, :, 0, 1].shape))
    T[:, :, :, 1, 0] = T[:, :, :, 0, 1]
    T[:, :, :, 0, 2] = dVdx * dVdz
    if noise > 0:
        T[:, :, :, 0, 2] += np.abs(np.random.normal(0.0, noise, T[:, :, :, 0, 2].shape))
    T[:, :, :, 2, 0] = T[:, :, :, 0, 2]
    T[:, :, :, 1, 2] = dVdy * dVdz
    if noise > 0:
        T[:, :, :, 1, 2] += np.abs(np.random.normal(0.0, noise, T[:, :, :, 1, 2].shape))
    T[:, :, :, 2, 1] = T[:, :, :, 1, 2]

    # if sigma is zero, no blurring
    if sigma > 0:
        kernel = np.ones((sigma, sigma, sigma, 1, 1))
        T = signal.convolve(T, kernel, mode="same") 
        #T = sp.ndimage.gaussian_filter(T, [sigma, sigma, sigma, 0, 0])         # my version
    
    if SAVE:
        np.save('tensorfield.npy', T.astype(np.float32))
    else:
        return T



def structure3d_nz(V, sigma=3):
    
    T = structure3d(V, sigma)
    nz = V == 0
    T[nz, :, :] = 0
    return T


def save_rgb_volume(vol, SAVE=False):
    """
    This function takes a 3D grayscale volume and saves the rgb version (uint8)
    for visualization in tensorview3D.

    Parameters:
    vol (uint8): the grayscale volume (ZxYxX)

    Returns:
    uint8: the corresponding RGB volume (3xZxYxX)
    """
    rgb_vol = np.stack((vol,) * 3, axis=-1)
    if SAVE: 
        np.save('rgb_vol.npy', rgb_vol)
    else:
        return rgb_vol
    


'''
if __name__ == "__main__":
    if(len(sys.argv) < 2):
        print("Provide an image file name for processing")
        exit()
    else:
        filename = sys.argv[1]
        print("Processing file " + filename)
        V = imread(filename)
        T = structure3d_nz(V, 0)
    
    if(len(sys.argv) > 2):
        # the user provided an output file name
        outfilename = sys.argv[2]
    else:
        outfilename = "output.npy"
        
    print("Saving tensor field as " + outfilename)
    np.save(outfilename, T.astype(np.float32))
'''  

