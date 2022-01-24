"""
Created by Mikhaylov Andrey. 
Last update 24.01.2022.

andrey.mikhaylov@kit.edu

"""

__version__ = '1.02_24.01.2022'


import numpy as np
# import matplotlib.pyplot as plt
# from skimage.util import random_noise
from scipy.signal import unit_impulse

def image_derivatives(image):
    """
    

    Parameters
    ----------
    image : ndarray of floats
        your image, scpeciffically absolute phase

    Returns
    -------
    Dx : ndarray of floats
        gradient in x direction\diff. phase in x direction.
    Dy : ndarray of floats
        gradient in y direction\diff. phase in y direction.

    """

    Dx = np.gradient(image, axis=1)
    Dy = np.gradient(image, axis=0)
    return Dx,Dy



def gauss_single_flat(n):
    sigma_x = 1
    sigma_y = 1
    center_x = np.floor(n/2)
    center_y = np.floor(n/2)
    height = 1
    order = 1
    x,y = np.meshgrid(np.linspace(0,n-1,n),np.linspace(0,n-1,n))
    gauss = np.ones((n,n))
    for i in range(gauss.shape[0]):
        for j in range(gauss.shape[1]):
            gauss[i,j] = height*np.exp(-( ((center_x-x[i,j])**2/(2*sigma_x**2)) + ((center_y-y[i,j])**2/(2*sigma_y**2)) )**order )
    
    # plt.imshow(gauss)
    # plt.colorbar()
    # plt.show()
    return gauss



            
def gauss_single_full(sigma_x,sigma_y,center_x,center_y,height,n,offset,order=1):
  x,y = np.meshgrid(np.linspace(0,n-1,n),np.linspace(0,n-1,n))
  gauss = np.ones((n,n))
  center_x +=np.floor(n/2)
  center_y +=np.floor(n/2)
  for i in range(gauss.shape[0]):
      for j in range(gauss.shape[1]):
          gauss[i,j] = height*np.exp(-( ((center_x-x[i,j])**2/(2*sigma_x**2)) + ((center_y-y[i,j])**2/(2*sigma_y**2)) )**order ) + offset
      
  # plt.imshow(gauss)
  # plt.colorbar()
  # plt.show()
  return gauss
    
def gauss_multiple_flat(n,N,noise=0):
    """
    simulates ideal gaussian pattern. 

    
    Parameters
    ----------
    n: int, preferably even to match FFT restrictions
        amount of pixels per single unit cell(one gaussian)
        
    N: int, preferably even to match FFT restrictions
        amount of gaussians per side
    
    noise: float
        random noise in the values of mean image value
        
    
    Returns
    -------
    array of size (N*n,N*n) with N*N gaussians, each sampled by n*n points
    
    """       
    gauss_full_flat = np.tile(gauss_single_flat(n), (N, N))
    if noise!=0:
        gauss_full_flat = gauss_full_flat + np.mean(gauss_full_flat)*noise*np.random.rand(N*n,N*n)
    # plt.imshow(gauss_full)
    # plt.colorbar()
    # plt.show()
    return gauss_full_flat
             
    
def gauss_multiple_full(sigma_x,sigma_y,center_x,center_y,height,n,N,offset,obsc_mask=None,noise=0):
    """
    simulates gaussian pattern. 
    
    for ideal case: 
            sigma_x,sigma_y,center_x,center_y,height = arrays of ones, offset = array of zeros, noise = 0.
            Or just call gauss_multiple_flat
    
    
    Parameters
    ----------
    
    sigma_x,sigma_y: arrays of size N*N
        gaussian broadening (used for scattering pseudo-simulation) 
   
    center_x,center_y: arrays of size N*N
        position of the gaussian center within single unit cell (one gaussian) (used for refraction pseudo-simulation)
    
    height: array of size N*N
        height of the gaussian (used for absorption pseudo-simulation)
    
    offset: arrays of size N*N
        offset of the gaussian, simulates constant background\flux
        
    n: int, preferably even to match FFT restrictions
        amount of pixels per single unit cell(one gaussian)
        
    N: int, preferably even to match FFT restrictions
        amount of gaussians per side
    
    obsc_mask: array of size N*N
        if wavefront has discontinuity, provide mask taht covers this region
    
    noise: float
        random noise in the values of mean image value
        
    
    Returns
    -------
    array of size (N*n,N*n) with N*N gaussians, each sampled by n*n points
    
    """   

    gauss_full = np.ones((N*n,N*n))
    for n_x in range(N):
        for n_y in range(N):
            if  obsc_mask !=None and obsc_mask[n_x,n_y] == 0:
                gauss_full[n_x*n:n+n_x*n,n_y*n:n+n_y*n] = 10*unit_impulse((n,n),idx='mid')
            else:
                g = gauss_single_full(sigma_x[n_x,n_y],sigma_y[n_x,n_y],center_x[n_x,n_y],center_y[n_x,n_y],height[n_x,n_y],n,offset[n_x,n_y],order=1)
                gauss_full[n_x*n:n+n_x*n,n_y*n:n+n_y*n] = g
    if noise!=0:
        gauss_full = gauss_full + np.mean(gauss_full)*noise*np.random.rand(N*n,N*n)
    # plt.imshow(gauss_full)
    # plt.colorbar()
    # plt.show()
    return gauss_full   