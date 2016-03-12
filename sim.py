import fft_scale
import func
import numpy as np
import matplotlib.pylab as plt
from numba import jit

class Grid(object):
	def __init__(self, pix_size_arcmin, pix_len, beamFWHM, Delta_T, ell, CTT, CEE, CPP):
		self.pix_size_arcmin = pix_size_arcmin
		self.pix_len	     = pix_len
		self.sky_size_arcmin = pix_size_arcmin * pix_len
		self.FWHM	     = beamFWHM
		self.Delta_T	     = Delta_T
		self.ell	     = ell
		self.CTT	     = CTT
		self.CEE	     = CEE
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
		self.angle	 = np.angle(self.k1 + 1.j*self.k2)

class Spectrum(object):
	def __init__(self, grid):
		beamSQ    = np.exp( (grid.k**2) * (grid.FWHM*np.pi/(180.*60.))**2/(8*np.log(2.)) )
		self.CNT  = ( grid.Delta_T*np.pi/(180.*60.) )**2 * beamSQ  #Delta_T (muk-arcmin --> muk-radian)
		
		logCTT    = np.interp(grid.k, grid.ell, np.log(grid.CTT), left=-np.inf)
		self.CTT  = np.exp(logCTT)

		logCEE    = np.interp(grid.k, grid.ell, np.log(abs(grid.CEE)), left=-np.inf)
		self.CEE  = np.exp(logCEE)

		logCPP    = np.interp(grid.k, grid.ell, np.log(grid.CPP), left=-np.inf)
		self.CPP  = np.exp(logCPP)

class TQUandFriends(object):
	def __init__(self, TK, Qk, Uk, grd):
		self.Tx = fft_scale.ifft2r(Tk, grd.deltx)	
		self.Qx = fft_scale.ifft2r(Qk, grd.deltx)	
		self.Ux = fft_scale.ifft2r(Uk, grd.deltx)	
	
		self.d1Tx = fft.scale.ifft2r( (0.+1.j)*grd.k1*Tk, grd.deltx )
		self.d2Tx = fft.scale.ifft2r( (0.+1.j)*grd.k2*Tk, grd.deltx )
		self.d1Qx = fft.scale.ifft2r( (0.+1.j)*grd.k1*Qk, grd.deltx )
		self.d2Qx = fft.scale.ifft2r( (0.+1.j)*grd.k2*Qk, grd.deltx )
		self.d1Ux = fft.scale.ifft2r( (0.+1.j)*grd.k1*Uk, grd.deltx )
		self.d2Ux = fft.scale.ifft2r( (0.+1.j)*grd.k2*Uk, grd.deltx )

		self.d11Tx = fft.scale.ifft2r(-1.*grd.k1* grd.k1*Tk, grd.deltx )
		self.d12Tx = fft.scale.ifft2r(-1.*grd.k1* grd.k2*Tk, grd.deltx )
		self.d22Tx = fft.scale.ifft2r(-1.*grd.k2* grd.k2*Tk, grd.deltx )
		self.d11Qx = fft.scale.ifft2r(-1.*grd.k1* grd.k1*Qk, grd.deltx )
		self.d12Qx = fft.scale.ifft2r(-1.*grd.k1* grd.k2*Qk, grd.deltx )
		self.d22Qx = fft.scale.ifft2r(-1.*grd.k2* grd.k2*Qk, grd.deltx )
		self.d11Ux = fft.scale.ifft2r(-1.*grd.k1* grd.k1*Uk, grd.deltx )
		self.d12Ux = fft.scale.ifft2r(-1.*grd.k1* grd.k2*Uk, grd.deltx )
		self.d22Ux = fft.scale.ifft2r(-1.*grd.k2* grd.k2*Uk, grd.deltx )

@jit
def snd_ord_lense(Tx, d1Tx, d2Tx, d11Tx, d12Tx, d22Tx, intx, inty, rdisplx, rdisply):
	M, N = Tx.shape
	lTx  = np.zeros((M,N))

	for i in range(M):
		for j in range(N):
			lTx   [i,j] 	= Tx   [intx[i,j], inty[i,j]]
			d1lTx [i,j] 	= d1Tx [intx[i,j], inty[i,j]]
			d2lTx [i,j] 	= d2Tx [intx[i,j], inty[i,j]]
			d11lTx[i,j]	= d11Tx[intx[i,j], inty[i,j]]
			d12lTx[i,j]	= d12Tx[intx[i,j], inty[i,j]]
			d22lTx[i,j]	= d22Tx[intx[i,j], inty[i,j]]

			lTx[i,j] += d1lTx[i,j] * rdisplx[i,j] + d2lTx[i,j] * rdisply[i,j]
			lTx[i,j] += 0.5 * (rdisplx[i,j] * d11lTx[i,j] * rdisplx[i,j])	
			lTx[i,j] +=       (rdisplx[i,j] * d12lTx[i,j] * rdisply[i,j])	
			lTx[i,j] += 0.5 * (rdisply[i,j] * d22lTx[i,j] * rdisply[i,j])	
	return lTx 
		

def dcplense(displx, disply, grd):
	grdx, grdy = np.meshgrid(np.arange(grd.pix_len), np.arange(grd.pix_len))

	ndisplx =  np.round(displx/grd.deltx).astype(int)	
	ndisply =  np.round(disply/grd.deltx).astype(int)	
	rdisplx =  displx - grd.deltx * ndisplx 
	rdisply =  disply - grd.delty * ndisply
	return grdx+ndisplx, grdy+ndisply, rdisplx, rdisply

def scd_ord_len_maps(grid, padportion):
	spec = Spectrum(grid)

	#noise
	znt  = np.random.randn(grid.pix_len, grid.pix_len)
	ntk  = np.fft.fft2(znt)*np.sqrt(spec.CNT)/grid.deltx

	intx, inty, rdisplx, rdisply = dcplense(phidx1, phidx1, grd)

	TQU     = TQUandFriends(Tk, Qk, Uk, grd)
	tildeTx = snd_ord_lense(TQU.Tx, TQU.d1Tx, TQU.d2Tx, TQU.d11Tx, TQU.d12Tx, TQU.d22Tx, intx, inty, rdisplx, rdisply)
	tildeQx = snd_ord_lense(TQU.Qx, TQU.d1Qx, TQU.d2Qx, TQU.d11Qx, TQU.d12Qx, TQU.d22Qx, intx, inty, rdisplx, rdisply)
	tildeUx = snd_ord_lense(TQU.Ux, TQU.d1Ux, TQU.d2Ux, TQU.d11Ux, TQU.d12Ux, TQU.d22Ux, intx, inty, rdisplx, rdisply)

	return 
	
def all_ord_len_maps(grid, pad_portion):
	spec = Spectrum(grid)

	#noise
	znt  = np.random.randn(grid.pix_len, grid.pix_len)
	ntk  = np.fft.fft2(znt)*np.sqrt(spec.CNT)/grid.deltx

	#high-res TQU and Phi for lensing
	hrfactor= 3
	gridhr  = Grid(grid.pix_size_arcmin/hrfactor, grid.pix_len*hrfactor, grid.FWHM, grid.Delta_T, grid.ell, grid.CTT, grid.CEE, grid.CPP)
	spechr  = Spectrum(gridhr)

	rd = ( gridhr.deltk/gridhr.deltx ) * np.random.randn( gridhr.pix_len, gridhr.pix_len )
	tk = fft_scale.fft2( rd, gridhr.deltx) * np.sqrt(spechr.CTT/gridhr.deltk**2)
	tx = fft_scale.ifft2r(tk, gridhr.deltx)		

	rd = ( gridhr.deltk/gridhr.deltx ) * np.random.randn( gridhr.pix_len, gridhr.pix_len )
	Ek = fft_scale.fft2( rd, gridhr.deltx) * np.sqrt(spechr.CEE/gridhr.deltk**2)
	Qk = - Ek*np.cos(2.*gridhr.angle) #+ Bk*np.sin(2.*gridhr.angle)
	Uk = - Ek*np.sin(2.*gridhr.angle) #- Bk*np.cos(2.*gridhr.angle)
	Qx = fft_scale.ifft2r(Qk, gridhr.deltx)
	Ux = fft_scale.ifft2r(Uk, gridhr.deltx)


	rd     = (gridhr.deltk/gridhr.deltx) * np.random.randn(gridhr.pix_len, gridhr.pix_len)
	phik   = fft_scale.fft2(rd, gridhr.deltx) * np.sqrt(spechr.CPP/gridhr.deltk**2)
	phix   = fft_scale.ifft2r(phik, gridhr.deltx)		
	phidx1 = fft_scale.ifft2r( (0.+1.j)* gridhr.k1* phik, gridhr.deltx )
	phidx2 = fft_scale.ifft2r( (0.+1.j)* gridhr.k2* phik, gridhr.deltx )


	plt.subplot(2,2,1)
	plt.imshow(phix, origin='lower', extent=[0, grid.sky_size_arcmin/60.,0, grid.sky_size_arcmin/60.])
	plt.colorbar(format='%.0e')
	plt.subplot(2,2,2)
	plt.imshow(tx, origin='lower', extent=[0, grid.sky_size_arcmin/60.,0, grid.sky_size_arcmin/60.])
	plt.colorbar()
	plt.subplot(2,2,3)
	plt.imshow(Qx[::hrfactor, ::hrfactor], origin='lower', extent=[0, grid.sky_size_arcmin/60.,0, grid.sky_size_arcmin/60.])
	plt.colorbar()
	plt.subplot(2,2,4)
	plt.imshow(Ux[::hrfactor, ::hrfactor], origin='lower', extent=[0, grid.sky_size_arcmin/60.,0, grid.sky_size_arcmin/60.])
	plt.colorbar()
	plt.show()

	print np.max(abs(phidx1/grid.deltx))
	print np.max(abs(phidx2/grid.deltx))

	#lensing T
	tildetx = func.spline_interp2(gridhr.x, gridhr.y, tx, gridhr.x+phidx1, gridhr.y+phidx2, pad_portion) 
	tildetk = fft_scale.fft2(tildetx[::hrfactor, ::hrfactor], grid.deltx)
	ytk     = tildetk + ntk

	#lensing Q and U
	tildeQx = func.spline_interp2(gridhr.x, gridhr.y, Qx, gridhr.x+phidx1, gridhr.y+phidx2, pad_portion) 
	tildeUx = func.spline_interp2(gridhr.x, gridhr.y, Ux, gridhr.x+phidx1, gridhr.y+phidx2, pad_portion) 

	plt.subplot(2,2,1)
	plt.imshow(phix[::hrfactor, ::hrfactor], origin='lower', extent=[0, grid.sky_size_arcmin/60.,0, grid.sky_size_arcmin/60.])
	plt.subplot(2,2,2)
	plt.imshow( (tildetx-tx)[::hrfactor, ::hrfactor], origin='lower', extent=[0, grid.sky_size_arcmin/60.,0, grid.sky_size_arcmin/60.])
	plt.subplot(2,2,3)
	plt.imshow( (tildeQx-Qx)[::hrfactor, ::hrfactor], origin='lower', extent=[0, grid.sky_size_arcmin/60.,0, grid.sky_size_arcmin/60.])
	plt.subplot(2,2,4)
	plt.imshow( (tildeUx-Ux)[::hrfactor, ::hrfactor], origin='lower', extent=[0, grid.sky_size_arcmin/60.,0, grid.sky_size_arcmin/60.])
	plt.show()

def setpar():
	ell, DTT, DEE, DTE, Cdd, CTd = np.loadtxt('camb/test_scalCls.dat').T		

	CTT  = DTT*(2.*np.pi)/(ell*(ell+1))
	CEE  = DEE*(2.*np.pi)/(ell*(ell+1))
	CPP  = Cdd/(7.4311e12*ell**4)
	grd  = Grid(2., 2**9, 1., 8., ell, CTT, CEE, CPP)

	maps(grd, 0.05)


setpar()
