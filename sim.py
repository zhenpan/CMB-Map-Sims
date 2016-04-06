import fft_scale
import func
import numpy as np
import matplotlib.pylab as plt
from numba import jit
from sts import Cl_est
from sts import Amp_avg

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
		self.k           = np.ceil(np.sqrt(self.k1**2+self.k2**2))
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
	def __init__(self, Tk, Qk, Uk, grd):
		self.Tx = fft_scale.ifft2r(Tk, grd.deltx)	
		self.Qx = fft_scale.ifft2r(Qk, grd.deltx)	
		self.Ux = fft_scale.ifft2r(Uk, grd.deltx)	
	
		self.d1Tx = fft_scale.ifft2r( (0.+1.j)*grd.k1*Tk, grd.deltx )
		self.d2Tx = fft_scale.ifft2r( (0.+1.j)*grd.k2*Tk, grd.deltx )
		self.d1Qx = fft_scale.ifft2r( (0.+1.j)*grd.k1*Qk, grd.deltx )
		self.d2Qx = fft_scale.ifft2r( (0.+1.j)*grd.k2*Qk, grd.deltx )
		self.d1Ux = fft_scale.ifft2r( (0.+1.j)*grd.k1*Uk, grd.deltx )
		self.d2Ux = fft_scale.ifft2r( (0.+1.j)*grd.k2*Uk, grd.deltx )

		self.d11Tx = fft_scale.ifft2r(-1.*grd.k1* grd.k1*Tk, grd.deltx )
		self.d12Tx = fft_scale.ifft2r(-1.*grd.k1* grd.k2*Tk, grd.deltx )
		self.d22Tx = fft_scale.ifft2r(-1.*grd.k2* grd.k2*Tk, grd.deltx )
		self.d11Qx = fft_scale.ifft2r(-1.*grd.k1* grd.k1*Qk, grd.deltx )
		self.d12Qx = fft_scale.ifft2r(-1.*grd.k1* grd.k2*Qk, grd.deltx )
		self.d22Qx = fft_scale.ifft2r(-1.*grd.k2* grd.k2*Qk, grd.deltx )
		self.d11Ux = fft_scale.ifft2r(-1.*grd.k1* grd.k1*Uk, grd.deltx )
		self.d12Ux = fft_scale.ifft2r(-1.*grd.k1* grd.k2*Uk, grd.deltx )
		self.d22Ux = fft_scale.ifft2r(-1.*grd.k2* grd.k2*Uk, grd.deltx )

@jit
def snd_ord_lense(tx, d1tx, d2tx, d11tx, d12tx, d22tx, intx, inty, rdisplx, rdisply):
	M, N 	= tx.shape
	ltx	= np.zeros((M,N))
	d1ltx	= np.zeros((M,N))
	d2ltx   = np.zeros((M,N))
	d11ltx  = np.zeros((M,N))
	d12ltx  = np.zeros((M,N))
	d22ltx  = np.zeros((M,N))

	for i in range(M):
		for j in range(N):
			ltx   [i,j] 	= tx   [inty[i,j], intx[i,j]] #be careful with the index order x-> col, y->row
			d1ltx [i,j] 	= d1tx [inty[i,j], intx[i,j]]
			d2ltx [i,j] 	= d2tx [inty[i,j], intx[i,j]]
			d11ltx[i,j]	= d11tx[inty[i,j], intx[i,j]]
			d12ltx[i,j]	= d12tx[inty[i,j], intx[i,j]]
			d22ltx[i,j]	= d22tx[inty[i,j], intx[i,j]]

			ltx[i,j] += d1ltx[i,j] * rdisplx[i,j] 
			ltx[i,j] += d2ltx[i,j] * rdisply[i,j]
			ltx[i,j] += 0.5 * (rdisplx[i,j] * d11ltx[i,j] * rdisplx[i,j])	
			ltx[i,j] +=       (rdisplx[i,j] * d12ltx[i,j] * rdisply[i,j])	
			ltx[i,j] += 0.5 * (rdisply[i,j] * d22ltx[i,j] * rdisply[i,j])	
	return ltx 
		

def dcplense(displx, disply, grd):
	grdx, grdy = np.meshgrid(np.arange(grd.pix_len), np.arange(grd.pix_len))

	ndisplx =  np.round(displx/grd.deltx).astype(int)	
	ndisply =  np.round(disply/grd.deltx).astype(int)	
	intx	= (grdx + ndisplx) % grd.pix_len  		#index wrap
	inty	= (grdy + ndisply) % grd.pix_len 
	rdisplx =  displx - ndisplx * grd.deltx  
	rdisply =  disply - ndisply * grd.deltx 
	return intx, inty, rdisplx, rdisply  

class scd_ord_len_maps(object):
	def __init__(self, grid, padportion):
		spec = Spectrum(grid)

		#noise
		znt  = np.random.randn(grid.pix_len, grid.pix_len)
		ntk  = np.fft.fft2(znt)*np.sqrt(spec.CNT)/grid.deltx
		ntx  = fft_scale.ifft2r(ntk, grid.deltx)

		rd      = ( grid.deltk/grid.deltx ) * np.random.randn( grid.pix_len, grid.pix_len )
		self.Tk = fft_scale.fft2(  rd, grid.deltx) * np.sqrt(spec.CTT/grid.deltk**2)
		self.Tx = fft_scale.ifft2r(self.Tk, grid.deltx)		
	
		rd 	= ( grid.deltk/grid.deltx ) * np.random.randn( grid.pix_len, grid.pix_len )
		self.Ek	= fft_scale.fft2( rd, grid.deltx) * np.sqrt(spec.CEE/grid.deltk**2)
		self.Qk	= -1.* self.Ek * np.cos(2.*grid.angle) #+ Bk*np.sin(2.*grid.angle)
		self.Uk	= -1.* self.Ek * np.sin(2.*grid.angle) #- Bk*np.cos(2.*grid.angle)
		self.Qx = fft_scale.ifft2r(self.Qk, grid.deltx)
		self.Ux = fft_scale.ifft2r(self.Uk, grid.deltx)

		rd    	    = (grid.deltk/grid.deltx) * np.random.randn(grid.pix_len, grid.pix_len)
		self.phik   = fft_scale.fft2(rd, grid.deltx) * np.sqrt(spec.CPP/grid.deltk**2)
		self.phix   = fft_scale.ifft2r(self.phik, grid.deltx)		
		phidx1 	    = fft_scale.ifft2r( (0.+1.j)* grid.k1* self.phik, grid.deltx )
		phidx2 	    = fft_scale.ifft2r( (0.+1.j)* grid.k2* self.phik, grid.deltx )

		intx, inty, rdisplx, rdisply = dcplense(phidx1, phidx2, grid)

		TQU        = TQUandFriends(self.Tk, self.Qk, self.Uk, grid)
		self.tldTx = snd_ord_lense(TQU.Tx, TQU.d1Tx, TQU.d2Tx, TQU.d11Tx, TQU.d12Tx, TQU.d22Tx, intx, inty, rdisplx, rdisply)
		self.tldQx = snd_ord_lense(TQU.Qx, TQU.d1Qx, TQU.d2Qx, TQU.d11Qx, TQU.d12Qx, TQU.d22Qx, intx, inty, rdisplx, rdisply)
		self.tldUx = snd_ord_lense(TQU.Ux, TQU.d1Ux, TQU.d2Ux, TQU.d11Ux, TQU.d12Ux, TQU.d22Ux, intx, inty, rdisplx, rdisply)

		self.tldTk = fft_scale.fft2(self.tldTx, grid.deltx) 
		self.tldQk = fft_scale.fft2(self.tldQx, grid.deltx)
		self.tldUk = fft_scale.fft2(self.tldUx, grid.deltx)

		self.tldEk = -1.* self.tldQk * np.cos(2.*grid.angle) -1.*self.tldUk*np.sin(2.*grid.angle)
		self.tldBk = 	  self.tldQk * np.sin(2.*grid.angle) -1.*self.tldUk*np.cos(2.*grid.angle)
	
	
class all_ord_len_maps(object):
	def __init__(self, grid, pad_portion):
		spec = Spectrum(grid)
	
		#noise
		znt  = np.random.randn(grid.pix_len, grid.pix_len)
		ntk  = np.fft.fft2(znt)*np.sqrt(spec.CNT)/grid.deltx

		#high-res TQU and Phi for lensing
		hrfactor= 3
		gridhr  = Grid(grid.pix_size_arcmin/hrfactor, grid.pix_len*hrfactor, grid.FWHM, grid.Delta_T, grid.ell, grid.CTT, grid.CEE, grid.CPP)
		spechr  = Spectrum(gridhr)

		rd = ( gridhr.deltk/gridhr.deltx ) * np.random.randn( gridhr.pix_len, gridhr.pix_len )
		Tk = fft_scale.fft2( rd, gridhr.deltx) * np.sqrt(spechr.CTT/gridhr.deltk**2)
		Tx = fft_scale.ifft2r(Tk, gridhr.deltx)		

		rd = ( gridhr.deltk/gridhr.deltx ) * np.random.randn( gridhr.pix_len, gridhr.pix_len )
		self.Ek = fft_scale.fft2( rd, gridhr.deltx) * np.sqrt(spechr.CEE/gridhr.deltk**2)
		Qk = - self.Ek*np.cos(2.*gridhr.angle) #+ Bk*np.sin(2.*gridhr.angle)
		Uk = - self.Ek*np.sin(2.*gridhr.angle) #- Bk*np.cos(2.*gridhr.angle)
		Qx = fft_scale.ifft2r(Qk, gridhr.deltx)
		Ux = fft_scale.ifft2r(Uk, gridhr.deltx)


		rd     = (gridhr.deltk/gridhr.deltx) * np.random.randn(gridhr.pix_len, gridhr.pix_len)
		phik   = fft_scale.fft2(rd, gridhr.deltx) * np.sqrt(spechr.CPP/gridhr.deltk**2)
		phix   = fft_scale.ifft2r(phik, gridhr.deltx)		
		phidx1 = fft_scale.ifft2r( (0.+1.j)* gridhr.k1* phik, gridhr.deltx )
		phidx2 = fft_scale.ifft2r( (0.+1.j)* gridhr.k2* phik, gridhr.deltx )


		#lensing TQU
		self.tldTx = func.spline_interp2(gridhr.x, gridhr.y, Tx, gridhr.x+phidx1, gridhr.y+phidx2, pad_portion) 
		self.tldQx = func.spline_interp2(gridhr.x, gridhr.y, Qx, gridhr.x+phidx1, gridhr.y+phidx2, pad_portion) 
		self.tldUx = func.spline_interp2(gridhr.x, gridhr.y, Ux, gridhr.x+phidx1, gridhr.y+phidx2, pad_portion) 

		self.tldTk = fft_scale.fft2(self.tldTx[::hrfactor, ::hrfactor], grid.deltx)
		self.tldQk = fft_scale.fft2(self.tldQx[::hrfactor, ::hrfactor], grid.deltx)
		self.tldUk = fft_scale.fft2(self.tldUx[::hrfactor, ::hrfactor], grid.deltx)

		#raw TQU	
		self.Tx	= Tx[::hrfactor, ::hrfactor]
		self.Qx	= Qx[::hrfactor, ::hrfactor]
		self.Ux	= Ux[::hrfactor, ::hrfactor]

		self.Tk = fft_scale.fft2(self.Tx, grid.deltx)
		self.Qk = fft_scale.fft2(self.Qx, grid.deltx)
		self.Uk = fft_scale.fft2(self.Ux, grid.deltx)

def conplot(ax, bx, cx, dx, grid):
	plt.subplot(2,2,1)
	plt.imshow(ax, origin='lower', extent=[0, grid.sky_size_arcmin/60.,0, grid.sky_size_arcmin/60.])
	plt.colorbar(format='%.0e')
	plt.subplot(2,2,2)
	plt.imshow(bx, origin='lower', extent=[0, grid.sky_size_arcmin/60.,0, grid.sky_size_arcmin/60.])
	plt.colorbar()
	plt.subplot(2,2,3)
	plt.imshow(cx, origin='lower', extent=[0, grid.sky_size_arcmin/60.,0, grid.sky_size_arcmin/60.])
	plt.colorbar()
	plt.subplot(2,2,4)
	plt.imshow(dx, origin='lower', extent=[0, grid.sky_size_arcmin/60.,0, grid.sky_size_arcmin/60.])
	plt.colorbar()
	plt.show()


def setpar():
	ell, DTT, DEE, DTE, Cdd, CTd = np.loadtxt('camb/test_scalCls.dat').T		
	elltld, DTTtld, DEEtld, DBBtld, DTEtld = np.loadtxt('camb/test_lensedCls.dat').T

	CTT  = DTT*(2.*np.pi)/(ell*(ell+1))
	CEE  = DEE*(2.*np.pi)/(ell*(ell+1))
	CPP  = Cdd/(7.4311e12*ell**4)
	grd  = Grid(2., 2**9, 0., 8., ell, CTT, CEE, CPP)

	lmax = 6000	

	tt_avg 	  = Amp_avg()
	ee_avg 	  = Amp_avg()
	tldtt_avg = Amp_avg()
	tldee_avg = Amp_avg()
	tldbb_avg = Amp_avg()

	for i in range(2):
		print i 
		maps  = scd_ord_len_maps(grd, 0.05)
		rawtt = Cl_est(maps.Tk, grd, lmax, 30)
		rawee = Cl_est(maps.Ek, grd, lmax, 30)

		tldtt = Cl_est(maps.tldTk, grd, lmax, 30)
		tldee = Cl_est(maps.tldEk, grd, lmax, 30)
		tldbb = Cl_est(maps.tldBk, grd, lmax, 30)

		tt_avg.add(rawtt.cl); tldtt_avg.add(tldtt.cl)
		ee_avg.add(rawee.cl); tldee_avg.add(tldee.cl)
		tldbb_avg.add(tldbb.cl)

	conplot(maps.phix, maps.tldTx-maps.Tx, maps.Tx, maps.tldTx, grd)
	conplot(maps.phix, maps.tldTx-maps.Tx, maps.tldQx-maps.Qx, maps.tldUx-maps.Ux, grd)

	dell = lambda x: x*(x+1.)/(2.*np.pi)
	plt.loglog(rawtt.cbins, dell(rawtt.cbins)*tt_avg.avg, 'r.') #, label=r'$TT$') 
	plt.loglog(tldtt.cbins, dell(tldtt.cbins)*tldtt_avg.avg,'k.') #, label=r'$lensed \ TT$') 
	plt.loglog(rawee.cbins, dell(rawee.cbins)*ee_avg.avg, 'm.') #, label=r'$EE$') 
	plt.loglog(tldee.cbins, dell(tldee.cbins)*tldee_avg.avg,'b.') #, label=r'$lensed \ EE$') 
	plt.loglog(tldbb.cbins, dell(tldbb.cbins)*tldbb_avg.avg,'y.') #, label=r'$lensed \ BB$') 

	plt.loglog(ell, DTT, 'r-', label = r'$TT$')
	plt.loglog(elltld, DTTtld, 'k-', label = r'$lensed \ TT$')
	plt.loglog(ell, DEE, 'm-', label = r'$EE$')
	plt.loglog(elltld, DEEtld, 'b-', label = r'$lensed \ EE$')
	plt.loglog(elltld, DBBtld, 'y-', label = r'$lensed \ BB$')
	plt.legend(loc='best')
	plt.xlim(10, 6000)
	plt.show()

setpar()
