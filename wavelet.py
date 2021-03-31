import numpy as np
from check_data_compatibility import *

class Wavelet:
	
	"""
	This is a class of evaluating the wavelet kernel function.

	Attributes
	----------
	data : numpy.ndarray
		The array at which the wavelet kernel function is to evaluated.
	N, d : int, int
		The shape of data array.
	bw : float
		The bandwidth parameter in the wavelet kernel function; must be strictly positive.
	c : float
		The coefficient inside the cosine function; must be non-negative.

	Methods
	-------
	kernel_matrix(new_data)
		Evaluates the wavelet kernel function at new_data.

	kernel_single_eval(new_data)
		Evaluates the wavelet kernel function at new_data involving double `for` loops;
		less efficient than self.kernel_matrix.
	
	Reference
	---------
	Zhang, Li, Weida Zhou, and Licheng Jiao. 2004. “Wavelet Support Vector Machine.”
		IEEE Transactions on Systems, Man, and Cybernetics. Part B, Cybernetics:
		A Publication of the IEEE Systems, Man, and Cybernetics Society 34 (1): 34–39.

	"""
	
	def __init__(self, data, bw, c=1.75):
		
		"""
		Parameters
		----------
		data : numpy.ndarray
			The array at which the XXXXX kernel function is to evaluated.
		bw : float
			The bandwidth parameter in the wavelet kernel function; must be strictly positive.
		c : float, optional
			The coefficient inside the cosine function; must be non-negative.
			Default is 1.75.
		
		"""
		
		if not isinstance(data, np.ndarray):
			data = np.array(data)
		
		if len(data.shape) == 1:
			data = data.reshape(-1, 1)
		
		self.data = data
		self.N, self.d = self.data.shape
		if bw <= 0.:
			raise ValueError(f'The bandwidth parameter, bw, must be strictly positive.')
		else:
			self.bw = bw
		
		if c <= 0.:
			raise ValueError(f'The coefficient, c, must be non-negative.')
		else:
			self.c = c
	
	def kernel_matrix(self, new_data):
		
		"""
		Evaluates the wavelet kernel function at new_data.
		Each entry is k(X_i, Y_j), where X_i corresponds to the i-th row of data,
		Y_j corresponds to the j-th row of new_data.

		Parameters
		----------
		new_data : numpy.ndarray
			The array at which the wavelet kernel function is to evaluated.

		Returns
		-------
		numpy.ndarray
			The array of wavelet kernel function evaluations.

		"""
		
		if not isinstance(new_data, np.ndarray):
			new_data = np.array(new_data)
		
		if len(new_data.shape) == 1:
			new_data = new_data.reshape(-1, 1)
		
		n, d1 = new_data.shape
		
		check_data_compatibility(self.data, new_data)
		
		bw = self.bw
		
		tiled_data = np.tile(new_data, self.N).reshape(1, -1)
		tiled_land = np.tile(self.data.reshape(1, -1), n)
		
		# exponential part
		diff1 = - (tiled_data - tiled_land) ** 2 / (2 * bw ** 2)
		exp_power = np.sum(np.vstack(np.split(diff1, self.N * n, axis=1)), axis=1)
		exp_power = exp_power.reshape(n, self.N)
		exp_part = np.exp(exp_power)
		
		# cosine part
		diff2 = np.cos(self.c * (tiled_data - tiled_land) / self.bw)
		cos_part = np.prod(np.vstack(np.split(diff2, self.N * n, axis=1)), axis=1)
		cos_part = cos_part.reshape(n, self.N)
		
		output = exp_part * cos_part
		
		return output.T
	
	def kernel_single_eval(self, new_data):
		
		"""
		Evaluates the wavelet kernel function at new_data involving double `for` loops.
		This approach is less efficient comparing to self.kernel_matrix.
		Each entry is k(X_i, Y_j), where X_i corresponds to the i-th row of data,
		Y_j corresponds to the j-th row of new_data.

		Parameters
		----------
		new_data : numpy.ndarray
			The array at which the wavelet kernel function is to evaluated.

		Returns
		-------
		numpy.ndarray
			The array of wavelet kernel function evaluations.

		"""
		
		if not isinstance(new_data, np.ndarray):
			new_data = np.array(new_data)
		
		if len(new_data.shape) == 1:
			new_data = new_data.reshape(-1, 1)
			
		n, d1 = new_data.shape
		
		check_data_compatibility(self.data, new_data)
		
		output = np.zeros((self.N, n), dtype=np.float64)
		
		for i in range(self.N):
			for j in range(n):
				part1 = np.prod(np.cos(self.c * (self.data[i] - new_data[j]) / self.bw))
				part2 = np.exp(- np.sum((self.data[i] - new_data[j]) ** 2) / (2 * self.bw ** 2))
				output[i][j] = part1 * part2
		
		return output


if __name__ == '__main__':
	
	data1 = np.random.randn(500 * 3).reshape(500, 3)
	new_data1 = np.random.randn(1000 * 3).reshape(1000, 3)
	kernel = Wavelet(data1, 1.2)
	k1 = kernel.kernel_matrix(new_data1)
	k2 = kernel.kernel_single_eval(new_data1)
	print(np.allclose(k1, k2))
	
	data2 = np.random.randn(500)
	new_data2 = np.random.randn(1000)
	kernel = Wavelet(data2, 1.2)
	k3 = kernel.kernel_matrix(new_data2)
	k4 = kernel.kernel_single_eval(new_data2)
	print(np.allclose(k3, k4))
