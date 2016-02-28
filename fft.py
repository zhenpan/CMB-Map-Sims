
def fft2(fx, deltx):
	c  = deltx/(2.*np.pi)
	fk = np.fft2(fx)
	return fk*c

def ifft2(fk, deltk):
	c  = deltk/(2*np.pi)
	fx = np.ifft(fk)
	return fx*c

def ifft2r(fk, deltk):
	c  = deltk/(2*np.pi)
	fx = real(ifft(fk))
	return fx*c

def rft(fx, grid):
	c  = grid.deltx/(2.*np.pi) 
	fk = rfft(fx)
	return fk*c

def irft(fk, grid):
	c  = grid.deltk/(2.*np.pi)
	fx = irfft(ft)
	return fx
