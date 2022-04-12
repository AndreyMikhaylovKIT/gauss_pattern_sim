"""
Created by Mikhaylov Andrey. 
Last update 12.04.2022.

andrey.mikhaylov@kit.edu

"""

__version__ = '1.1_12.04.2022'


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


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np
    from skimage.morphology import disk
    from tifffile import imread, imsave
    from scipy import constants as const
    from skimage.transform import rotate
    from skimage.transform import rescale
    import os
    

    
    
    n = 16
    N = 128
    pix_size = 7.4e-6 #
    pitch = n*pix_size
    FoV = pitch*N
    E = 10007.5039 #energy in ev
    lambd = const.h * const.c / (E*const.e)
    beta = 3.70246567E-09
    delta = 2.66748725E-06
    mu = 4*const.pi * beta / lambd
    diameter = 0.999e-3
    radius_in_pixels = int(diameter/(2*pix_size))
    rot_angle = 25
    save = True
    path_to_save = r'D:\Users\np3733\Documents\Scripts_1\tomography_article\save'
    
    print('Amount of lenses per side is {}. Pixel size is {} um. Pitch of the SHSX is {} um. FoV is {} mm per side. Energy of X-rays is {} keV. Material is PMMA. Diameter of the rod is {} mm.'.format(N, pix_size*1e6,pitch*1e6,FoV*1000,E/1000,2))
    
    
    # creating flat gaussian pattern
    g = gauss_multiple_flat(n,N)

    plt.imshow(g)
    plt.colorbar()
    plt.title('flat pattern')
    plt.show()
    
    # creating slice of the 3D object, voxel size == pitch**3 and line projection of the slice
    slice_3d_object = np.pad(disk(radius_in_pixels),(956, 957), 'constant', constant_values=(0, 0))
    line_proj = np.sum(slice_3d_object,axis=0)
    
    # stacking line projections to create 2d projection and translating values to micrometers. rotation on rot_angle value
    
    proj = rotate(np.tile(line_proj,(2048,1)) * pix_size,rot_angle)
    
    
    
    plt.imshow(proj)
    plt.colorbar()
    plt.title('thickness projection')
    plt.show() 
    

    #  based on the thickness calculating transmission (T = exp(-mu*d)) and phase (Ф = 2*Pi*Delta*d/lambda)
    transmission = np.exp(-mu*proj)
    phase = 2 * const.pi * proj * delta / lambd
    
    
    plt.imshow(transmission)
    plt.colorbar()
    plt.title('transmission')
    plt.show() 
    
    plt.imshow(phase)
    plt.colorbar()
    plt.title('phase')
    plt.show() 
    
    # taking gradients to create differential phase contrasts
    dpcx,dpcy = image_derivatives(phase)
    

    # assuming object without scattering and attenating approx. 50% in the middle, creating corresponding gaussian pattern
    
    sigma_x = np.ones((N,N))
    sigma_y = np.ones((N,N))
    offset = np.zeros((N,N))
    
    height = np.copy(rescale(transmission,0.0625))
    center_x = rescale(dpcx,0.0625)
    center_y = rescale(dpcy,0.0625)                
    
    g_obj = gauss_multiple_full(sigma_x,sigma_y,center_x,center_y,height,n,N,offset)
    
    plt.imshow(g_obj,vmin=0,vmax=1)
    plt.colorbar()
    plt.title('obj pattern')
    plt.show() 
    
    
    if save:
        print('Results will be saved to ' + path_to_save)
        imsave(os.path.join(path_to_save,'transmission.tif'), transmission.astype(np.float32))
        imsave(os.path.join(path_to_save,'phase.tif'), phase.astype(np.float32))
        imsave(os.path.join(path_to_save,'proj.tif'), proj.astype(np.float32))
        imsave(os.path.join(path_to_save,'gauss_ref.tif'), g.astype(np.float32))
        imsave(os.path.join(path_to_save,'gauss_object.tif'), g_obj.astype(np.float32))
    else:
        print('Results will not be saved')
    

    
    
    
