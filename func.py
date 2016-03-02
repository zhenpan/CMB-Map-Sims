import numpy as np
import matplotlib.pylab as plt

def perodic_padd(x, y, z, pad_portion = 0.05):
	old_len = len(x[0,:])
	add_len = np.int(np.around(old_len*pad_portion))
	index   = np.arange(-add_len, old_len+add_len)
	deltx   = x[0,1]-x[0,0]
	
	full_x, full_y = np.meshgrid(deltx*index, deltx*index)
	full_z  = np.lib.pad(z, ((add_len, add_len),(add_len, add_len)), 'symmetric')	
	return full_z

x, y = np.meshgrid(np.arange(100), np.arange(100))
z = np.sin(x+y)
zd= perodic_padd(x,y,z, 0.05)


plt.imshow(zd)
plt.show()
