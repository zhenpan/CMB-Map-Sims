import numpy as np

def fft2(fx, deltx):
	c  = deltx**2/(2.*np.pi)
	fk = np.fft.fft2(fx)
	return fk*c

def ifft2(fk, deltx):
	c  = deltx**2/(2.*np.pi)
	fx = np.fft.ifft2(fk)
	return fx*1./c


def rft2(fx, deltx):
	c  = deltx**2/(2.*np.pi) 
	fk = np.fft.rfft2(fx)
	return fk*c

def irft2(fk, deltx):
	c  = deltx**2/(2.*np.pi)
	fx = np.fft.irfft2(fk)
	return fx*1./c


