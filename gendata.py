import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import image2tensor as st
import tensorvote as tv
import tensorvote3d as tv3
import skimage as ski
import os


def genSpiral3(N, d):
    """
    Generates a 3D volume containing a spiral shape.

    Parameters:
    N (int): The size of the 3D volume along each dimension (NxNxN).
    d (float): A scaling factor that controls the density and spread of the spiral within the volume.

    Returns:
    np.ndarray: A 3D numpy array of shape (N, N, N) with the spiral shape represented as ones (1.0) and the background as zeros.
    """  
    I = np.zeros((N, N, N)).astype(np.float32)
    
    dt = 0.00001
    t = 0
    
    while True:
        x = t * np.cos(t)
        y = t * np.sin(t)
        z = 2 * t
        
        xp = (x / (2 * d) + 0.5) * N
        yp = (y / (2 * d) + 0.5) * N
        zp = (z / (2 * d)) * N
        
        xi = int(xp)
        yi = int(yp)
        zi = int(zp)
        if(xi < 0 or yi < 0 or zi < 0 or xi >= N or yi >= N or zi >= N):
            return I
        
        I[zi, yi, xi] = 1.0
        
        t = t + dt
        
      
def saveColorStack(filename, I):
    """
    Saves a stack of 3D color images as individual files.

    Parameters:
    filename (str): The file name for the output files. Each file will have a numbered suffix.
    I (np.ndarray): A 4D numpy array representing the image stack. 
                    The shape is (depth, height, width, color_channels), with color as the last dimension.

    Returns:
    None: The function saves images to disk.
    """
    # get the number of images to save
    nz = I.shape[0]
    digits = len(str(nz))
    
    # uint conversion
    I8 = (I * 255).astype(np.uint8)
    
    for zi in range(nz):
        filestring = filename + "%0" + str(digits) + "d.bmp"
        ski.io.imsave(filestring %zi, I8[zi, :, :, :])
    

def saveGrayStack(filename, I):
    """
    Saves a stack of grayscale images as individual files.

    Parameters:
    filename (str): The base name for the output files. Each file will have a numbered suffix.
    I (np.ndarray): A 3D numpy array representing the image stack with shape (height, width, depth), 
                    where depth corresponds to the number of images.

    Returns:
    None: The function saves images to disk.
    """
    # get the number of images to save
    nz = I.shape[0]
    digits = len(str(nz))
    
    # uint conversion
    I8 = (I * 255).astype(np.uint8)
    
    for zi in range(nz):
        filestring = filename + "%0" + str(digits) + "d.bmp"
        ski.io.imsave(filestring %zi, I8[zi, :, :])


def genSample3D(shape, noise=0):
    """
    Creates a synthetic 3D volume containing various geometric shapes:
    a donut (torus), a straight tube, a curved tube, a hollow sphere, and a solid sphere.

    Parameters:
    shape (tuple of ints): The dimensions of the 3D volume, specified as (depth, height, width).
    noise (float, optional): Random noise to add to the volume. Default is 0 (no noise).

    Returns:
    np.ndarray: A 3D numpy array of the specified shape containing the generated shapes. 
                The array has values between 0 and 1, where 1 indicates the presence of a shape.
    """
    vol = np.zeros((shape));
    
    # for solid sphere
    center_solid = (shape[0] // 4, shape[1] // 4, shape[2] // 4)
    radius_solid = shape[0] // 10

    # for hollow sphere
    center_hollow = (shape[0] - shape[0] // 4, shape[1] - shape[1] // 4, shape[2] - shape[2] // 4)
    radius_hollow = shape[0] // 10
    thick_hollow = 2
    
    # for donut (torus shape)
    center_donut = (shape[0] // 4, shape[1] // 2, shape[2] - shape[2] // 4)
    radius_donut = shape[0] // 8
    thick_donut = 3
    
    # for curved tube (partial donut)
    center_curve = (shape[0] - shape[0] // 3, shape[1] // 8, -shape[2] // 6)
    radius_curve = shape[0] // 1.5
    thick_curve = 2
    
    # for straight tube
    radius_tube = 2
    
    t_values = np.linspace(0, 1, 100)  # Parameter range for smooth curve
    curve_x = (t_values) * (shape[0] // 2) + shape[0] // 2          # Starts at max x, curves to x=0
    curve_y = t_values * (shape[1] // 2) + shape[1] // 2            # Moves from y=0 to max y
    curve_z = t_values * (shape[2] // 2)                            # Moves from z=0 to max z
    
    for x in range(shape[0]):
        for y in range(shape[1]):
            for z in range(shape[2]):
                
                # solid sphere
                if np.sqrt((x - center_solid[0])**2 + (y - center_solid[1])**2 + (z - center_solid[2])**2) <= radius_solid:
                    vol[x, y, z] = 1
                
                # hollow sphere
                dist_hollow = np.sqrt((x - center_hollow[0])**2 + (y - center_hollow[1])**2 + (z - center_hollow[2])**2)
                if radius_hollow - thick_hollow <= dist_hollow <= radius_hollow:
                    vol[x, y, z] = 1
                    
                # donut
                distance_xy = np.sqrt((y - center_donut[1])**2 + (x - center_donut[0])**2)
                if (radius_donut - thick_donut) <= distance_xy <= (radius_donut + thick_donut):
                    # check the vertical distance from the center of the tube
                    if np.abs(z - center_donut[2]) <= thick_donut:
                        # check if the distance from the tube's center is correct
                        distance_donut = np.sqrt((distance_xy - radius_donut)**2 + (z - center_donut[2])**2)
                        if distance_donut <= thick_donut:
                            vol[x, y, z] = 1
                            
                # curved tube
                distance_yz = np.sqrt((y - center_curve[1])**2 + (z - center_curve[2])**2)
                if (radius_curve - thick_curve) <= distance_yz <= (radius_curve + thick_curve):
                    if np.abs(x - center_curve[0]) <= thick_curve:
                        distance_curve = np.sqrt((distance_yz - radius_curve)**2 + (x - center_curve[0])**2)
                        if distance_curve <= thick_curve:
                            vol[x, y, z] = 1
                
                # straight tube
                dist_tube = np.sqrt((x - curve_x)**2 + (y - curve_y)**2 + (z - curve_z)**2)
                if np.min(dist_tube) <= radius_tube:
                    vol[x, y, z] = 1

    if noise != 0:
        vol = np.random.normal(0, noise, vol.shape) + vol
    vol[vol<0] = 0
    return vol
