import numpy as np

class Amp_avg(object):
	def __init__(self):
		self.nsim 	= 0.
		self.av_sum	= 0.
		self.sq_sum	= 0.
	
	def add(self, amp):
		self.nsim	+= 1.
		self.av_sum	+= amp
		self.sq_sum	+= amp*amp
		self.avg	 = self.av_sum / self.nsim
		self.std	 = np.sqrt(self.sq_sum / self.nsim - self.avg**2) / np.sqrt(self.nsim) #std var of the mean of sims

class Cl_est(object):
	def __init__(self, tk, grd, lmax, dl=1):
		self.lmax= lmax
		self.dl	 = dl

		ell	= grd.k.flatten()
		self.nm, bins = np.histogram(ell, bins=np.arange(0, lmax+1, dl))
		self.cl, bins = np.histogram(ell, bins=np.arange(0, lmax+1, dl), weights = np.absolute(tk.flatten())**2)
		self.cl[np.nonzero(self.nm)] /= self.nm[np.nonzero(self.nm)]
		self.cl[np.nonzero(self.nm)] *= grd.deltk**2
		self.cbins    = 0.5*(bins[:-1] + bins[1:])
