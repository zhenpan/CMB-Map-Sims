import fft_scale
import func
import numpy as np
import matplotlib.pylab as plt


class Grid(object):
	def __init__(self, pix_size_arcmin, pix_len, beamFWHM, Delta_T, ell, CTT, CPP):
		self.pix_size_arcmin = pix_size_arcmin
		self.pix_len	     = pix_len
		self.sky_size_arcmin = pix_size_arcmin * pix_len
		self.FWHM	     = beamFWHM
		self.Delta_T	     = Delta_T
		self.ell	     = ell
		self.CTT	     = CTT
		self.CPP	     = CPP

		self.deltx     = pix_size_arcmin*np.pi/(180.*60.) #in radians
		self.x, self.y = np.meshgrid(np.arange(pix_len*1.), np.arange(pix_len*1.))
		self.x 	      *= self.deltx
		self.y 	      *= self.deltx

		self.period    = self.deltx * pix_len
		self.deltk     = 2*np.pi/self.period
		kside	       = (2*np.pi/self.deltx) * np.fft.fftfreq(pix_len)

		self.k1, self.k2 = np.meshgrid(kside, kside)
		self.k           = np.sqrt(self.k1**2+self.k2**2)

class Spectrum(object):
	def __init__(self, grid):
		beamSQ    = np.exp( (grid.k**2) * (grid.FWHM*np.pi/(180.*60.))**2/(8*np.log(2.)) )
		self.CNT  = ( grid.Delta_T*np.pi/(180.*60.) )**2 * beamSQ  #Delta_T (muk-arcmin --> muk-radian)
		
		logCTT    = np.interp(grid.k, grid.ell, np.log(grid.CTT), left=-np.inf)
		self.CTT  = np.exp(logCTT)

		logCPP    = np.interp(grid.k, grid.ell, np.log(grid.CPP), left=-np.inf)
		self.CPP  = np.exp(logCPP)

def maps(grid, pad_portion):
	spec = Spectrum(grid)
	znt  = np.random.randn(grid.pix_len, grid.pix_len)
	ntk  = np.fft.fft2(znt)*np.sqrt(spec.CNT)/grid.deltx

	hrfactor= 3
	gridhr  = Grid(grid.pix_size_arcmin/hrfactor, grid.pix_len*hrfactor, grid.FWHM, grid.Delta_T, grid.ell, grid.CTT, grid.CPP)
	spechr  = Spectrum(gridhr)

	zt = ( gridhr.deltk/gridhr.deltx ) * np.random.randn( gridhr.pix_len, gridhr.pix_len )
	tk = fft_scale.fft2( zt, gridhr.deltx) * np.sqrt(spechr.CTT/gridhr.deltk**2)
	tx = fft_scale.ifft2(tk, gridhr.deltx)		

	plt.imshow(tx.real, origin='lower', extent=[0, grid.sky_size_arcmin/60.,0, grid.sky_size_arcmin/60.])
	plt.colorbar()
	plt.show()

	zp     = (gridhr.deltk/gridhr.deltx) * np.random.randn(gridhr.pix_len, gridhr.pix_len)
	phik   = fft_scale.fft2(zp, gridhr.deltx) * np.sqrt(spechr.CPP/gridhr.deltk**2)
	phix   = fft_scale.ifft2(phik, gridhr.deltx)		
	phidx1 = fft_scale.ifft2( (0.+1.j)* gridhr.k1* phik, gridhr.deltx )
	phidx2 = fft_scale.ifft2( (0.+1.j)* gridhr.k2* phik, gridhr.deltx )

	tx     = tx.real
	phix   = phix.real
	phidx1 = phidx1.real
	phidx2 = phidx2.real

	print np.max(abs(phidx1/grid.deltx))
	print np.max(abs(phidx2/grid.deltx))

	tildetx = func.spline_interp2(gridhr.x, gridhr.y, tx, gridhr.x+phidx1, gridhr.y+phidx2, pad_portion) 
	tildetk = fft_scale.fft2(tildetx[::hrfactor, ::hrfactor], grid.deltx)
	ytk     = tildetk + ntk

	plt.subplot(1,2,1)
	plt.imshow(phix, origin='lower', extent=[0, grid.sky_size_arcmin/60.,0, grid.sky_size_arcmin/60.])
	plt.subplot(1,2,2)
	plt.imshow(tildetx-tx, origin='lower', extent=[0, grid.sky_size_arcmin/60.,0, grid.sky_size_arcmin/60.])
	plt.show()

def setpar():
	ell, DTT, DEE, DTE, Cdd, CTd = np.loadtxt('camb/test_scalCls.dat').T		

	CTT  = DTT*(2.*np.pi)/(ell*(ell+1))
	CPP  = Cdd/(7.4311e12*ell**4)
	grd  = Grid(2., 2**9, 1., 8., ell, CTT, CPP)

	maps(grd, 0.05)


	


setpar()
