import numpy as np
import matplotlib.pylab as plt
from scipy import interpolate

def perodic_padd(x, y, z, pad_portion = 0.05):
	old_len = len(x[0,:])
	add_len = np.int(np.around(old_len*pad_portion))
	index   = np.arange(-add_len, old_len+add_len)
	deltx   = x[0,1]-x[0,0]
	
	full_x, full_y = np.meshgrid(deltx*index, deltx*index)
	full_z  = np.lib.pad(z, ((add_len, add_len),(add_len, add_len)), 'symmetric')	
	return full_x, full_y, full_z

def spline_interp2(xtmp, ytmp, ztmp, xi, yi, pad_portion = 0.05): # xtmp, ytmp and xi, yi are meshgrids
	x, y, z = perodic_padd(xtmp, ytmp, ztmp, pad_portion)
	iz      = interpolate.RectBivariateSpline(y[:,0], x[0,:], z)
	zi      = iz.ev(yi, xi)
	return zi

