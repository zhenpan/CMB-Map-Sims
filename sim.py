import fft_scale
import func
import numpy as np
import matplotlib.pylab as plt


class Grid(object):
	def __init__(self, pix_size_arcmin, pix_len):
		self.pix_size_arcmin = pix_size_arcmin
		self.pix_len	     = pix_len
		self.sky_size_arcmin = pix_size_arcmin*pix_len

		self.deltx     = pix_size_arcmin*np.pi/(180.*60.) #in radians
		self.x, self.y = np.meshgrid(np.arange(pix_len*1.), np.arange(pix_len*1.))
		self.x 	      *= self.deltx
		self.y 	      *= self.deltx

		self.period    = self.deltx * self.pix_len
		self.deltk     = 2*np.pi/self.period
		kside	       = (2*np.pi/self.deltx) * np.fft.fftfreq(pix_len)

		self.k1, self.k2 = np.meshgrid(kside, kside)
		self.k           = np.sqrt(self.k1**2+self.k2**2)


class Spectrum(object):
	def __init__(self, pix_size_arcmin, pix_len, beamFWHM, Delta_T,  ell, CTT, CPP):
		self.Grid = Grid(pix_size_arcmin, pix_len)

		beamSQ    = np.exp( (self.Grid.k**2) * (beamFWHM*np.pi/(180.*60.))**2/(8*np.log(2.)) )
		self.CNT  = ( Delta_T*np.pi/(180.*60.) )**2 * beamSQ  #Delta_T (muk-arcmin --> muk-radian)
		
		logCTT    = np.interp(self.Grid.k, ell, np.log(CTT), left=-np.inf)
		self.CTT  = np.exp(logCTT)

		logCPP    = np.interp(self.Grid.k, ell, np.log(CPP), left=-np.inf)
		self.CPP  = np.exp(logCPP)

class maps(object):
	def __init__(self, spec, pad_portion):
		znt= np.random.randn(spec.Grid.pix_len, spec.Grid.pix_len)
		ntk= np.fft.fft2(znt)*np.sqrt(spec.CNT)/spec.Grid.deltx

		zt = np.random.randn(spec.Grid.pix_len, spec.Grid.pix_len)
		tk = np.fft.fft2(zt)*np.sqrt(spec.CTT)/spec.Grid.deltx
		tx = np.fft.ifft2(tk)		

		zp     = np.random.randn(spec.Grid.pix_len, spec.Grid.pix_len)
		phik   = np.fft.fft2(zp)*np.sqrt(spec.CPP)/spec.Grid.deltx
		phix   = np.fft.ifft2(phik)		
		phidx1 = fft_scale.ifft2( (0.+1.j)* spec.Grid.k1* phik, spec.Grid.deltx )
		phidx2 = fft_scale.ifft2( (0.+1.j)* spec.Grid.k2* phik, spec.Grid.deltx )

		self.tx     = tx.real
		self.phix   = phix.real
		self.phidx1 = phidx1.real
		self.phidx2 = phidx2.real

		print np.max(abs(self.phidx1/spec.Grid.deltx))
		print np.max(abs(self.phidx2/spec.Grid.deltx))

		self.tildetx = func.spline_interp2(spec.Grid.x, spec.Grid.y, self.tx, spec.Grid.x+self.phidx1, spec.Grid.y+self.phidx2, pad_portion) 
		self.tildetk = fft_scale.fft2(self.tildetx, spec.Grid.deltx)
		self.ytk     = self.tildetk + ntk

		plt.imshow(self.tildetx, origin='lower', extent=[0, spec.Grid.sky_size_arcmin/60.,0, spec.Grid.sky_size_arcmin/60.])
		plt.colorbar()
		plt.show()

def setpar():
	ell, DTT, DEE, DTE, Cdd, CTd = np.loadtxt('camb/test_scalCls.dat').T		

	CTT  = DTT*(2.*np.pi)/(ell*(ell+1))
	CPP  = Cdd/(7.4311e12*ell**4)

	spec = Spectrum(2., 2**9, 1., 8., ell, CTT, CPP)
	Tmap = maps(spec, 0.05)


	


setpar()
