import numpy as np
import scipy.optimize as opt
# import astropy.modeling.rotations as rot

a_primary = np.array([[0., 0., -57.74022, 1.5373825, 1.154294, -0.441762, 0.0906601,],
             [0., 0., 0., 0., 0., 0., 0.],
             [-72.17349, 1.8691899, 2.8859421, -1.026471, 0.2610568, 0., 0.],
             [0., 0., 0., 0., 0., 0., 0.],
             [1.8083973, -0.603195, 0.2177414, 0., 0., 0., 0.],
             [0., 0., 0., 0., 0., 0., 0.],
             [0.0394559, 0., 0., 0., 0., 0., 0.,]])

a_secondary = np.array([[0., 0., 103.90461, 6.6513025, 2.8405781, -0.7819705, -0.0400483,  0.0896645],
             [0., 0., 0., 0., 0., 0., 0., 0.],
             [115.44758, 7.3024355, 5.7640389, -1.578144, -0.0354326, 0.2781226, 0., 0.],
             [0., 0., 0., 0., 0., 0., 0., 0.],
             [2.9130983, -0.8104051, -0.0185283, 0.2626023, 0., 0., 0., 0.],
             [0., 0., 0., 0., 0., 0., 0., 0.],
             [-0.0250794, 0.0709672, 0., 0., 0., 0., 0., 0.,],
             [0., 0., 0., 0., 0., 0., 0., 0.]])

def mirror(x, y, a):
    z = 0.
    Rn = 3000.
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            z += a[i,j]*(x/Rn)**i * (y/Rn)**j
    return z

def secondary_fit_func(xy, x0, y0, z0): #, a1, a2, a3):
    z =  mirror(xy[0] - x0, xy[1] - y0, a_secondary) - z0
    # model = rot.RotationSequence3D([a1, a2, a3], axes_order='xyz')
    # x, y, z = model(xy[0], xy[1], z)
    return z

def mirror_fit(x, y, z, fit_func, **kwargs):
    popt, pcov = opt.curve_fit(fit_func, (x, y), z, **kwargs)
    z_fit = fit_func((x, y), *popt)
    rms = np.sqrt(np.mean((z - z_fit)**2))
    return popt, rms
