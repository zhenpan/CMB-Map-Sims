import numpy as np
import matplotlib.pylab as plt


class Grid(object):
	def __init__(self, pix_size_arcmin, pix_len):
		self.pix_size_arcmin = pix_size_arcmin
		self.pix_len = pix_len

		self.deltx = pix_size_arcmin*np.pi/(180.*60.) #in radians
		self.x, self.y = np.meshgrid(np.arange(pix_len*1.), np.arange(pix_len*1.))
		self.x *= self.deltx
		self.y *= self.deltx

		kside	= (2*np.pi/self.deltx) * np.fft.fftfreq(pix_len)
		self.k1, self.k2 = np.meshgrid(kside, kside)
		self.period  = self.deltx * pix_len
		self.deltk   = 2*np.pi/self.period
		self.r       = np.sqrt(self.k1**2+self.k2**2)


class Spectrum(object):
	def __init__(self, pix_size_arcmin, pix_len, beamFWHM, Delta_T,  ell, CTT):
		self.Grid = Grid(pix_size_arcmin, pix_len)
		beamSQ    = np.exp( (self.Grid.k1**2+self.Grid.k2**2)* (beamFWHM*np.pi/(180.*60.))**2/(8*np.log(2.)) )
		self.cNT  = ( Delta_T*np.pi/(180.*60.) )**2 * beamSQ  #Delta_T (muk-arcmin)
		
		logCTT 	  = np.interp(self.Grid.r, ell, np.log(CTT))
		logCTT[0][0] = - np.inf
		self.CTT  = np.exp(logCTT)

class maps(object):
	def __init__(self, spec):
		Tlm = np.zeros((spec.Grid.pix_len, spec.Grid.pix_len))
		for i in range(spec.Grid.pix_len):
			for j in range(spec.Grid.pix_len):
				if(spec.CTT[i][j] > 0.):
					Tlm[i][j] = np.random.normal(0., spec.CTT[i][j], 1)	 
				else:
					Tlm[i][j] = 0.

		self.Tmap = np.fft.ifft2(Tlm)
		plt.imshow(np.abs(self.Tmap))
		plt.colorbar()
		plt.show()

def setpar():
	ell, DTT, DEE, DTE, Cdd, CTd = np.loadtxt('camb/test_scalCls.dat').T		

	CTT = DTT*(2.*np.pi)/(ell*(ell+1))

	spec = Spectrum(2., 100, 1., 8., ell, CTT)
	Tmap = maps(spec)

#	plt.plot(index, DTT2d, '.')
#	plt.plot(ell, DTT)
#	plt.show()

	


setpar()
