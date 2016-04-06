cimport numpy as np

def primes(np.ndarray[np.float64_t, ndim=2] A):
	cdef int i, j 
	for i in range(A.shape[0]):
		for j in range(A.shape[1]):
			A[i,j] = i+j
