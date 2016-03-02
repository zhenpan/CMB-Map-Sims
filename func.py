import numpy as np

def perodic_padd(x, y, z, pad_portion = 0.05):
	old_len = len(x[0,:])
	add_len = np.around(len_old*pad_portion)
	index   = np.arange(-add_len, old_len+add_len)
	deltx   = x[0,1]-x[0,0]
	
	full_x, full_y = np.meshgrid(deltx*index, deltx*index)
	full_z  = np.lib.pad(ax, ((add_len, add_len),(add_len, add_len)), 'symmetric')	
