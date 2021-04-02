import numpy as np
from check_data_compatibility import *

class ChiSquare:

	"""
	This is a class of evaluating the chi-square kernel function.

	Attributes
	----------
	data : numpy.ndarray
		The array at which the chi-square kernel function is to evaluated.
	N, d : int, int
		The shape of data array.

	Methods
	-------
	kernel_matrix(new_data)
		Evaluates the chi-square kernel function at new_data.

	kernel_single_eval(new_data)
		Evaluates the chi-square kernel function at new_data involving double `for` loops;
		less efficient than self.kernel_matrix.
	
	Reference
	---------
	Vedaldi, Andrea, and Andrew Zisserman. 2012. “Efficient Additive Kernels via Explicit Feature Maps.”
		IEEE Transactions on Pattern Analysis and Machine Intelligence 34 (3): 480–92.

	"""
	
	def __init__(self, data):
		
		"""
		Parameters
		----------
		data : numpy.ndarray
			The array at which the chi-square kernel function is to evaluated.
		
		"""
		
		if not isinstance(data, np.ndarray):
			data = np.array(data)
		
		if len(data.shape) == 1:
			data = data.reshape(-1, 1)
		
		self.data = data
		self.N, self.d = self.data.shape
	
	def kernel_matrix(self, new_data):
		
		"""
		Evaluates the chi-square kernel function at new_data.
		Each entry is k(X_i, Y_j), where X_i corresponds to the i-th row of data,
		Y_j corresponds to the j-th row of new_data.

		Parameters
		----------
		new_data : numpy.ndarray
			The array at which the chi-square kernel function is to evaluated.

		Returns
		-------
		numpy.ndarray
			The array of chi-square kernel function evaluations.

		"""
		
		if not isinstance(new_data, np.ndarray):
			new_data = np.array(new_data)
		
		if len(new_data.shape) == 1:
			new_data = new_data.reshape(-1, 1)
		
		n, d1 = new_data.shape
		
		check_data_compatibility(self.data, new_data)
		
		tiled_data = np.tile(new_data, self.N).reshape(1, -1)
		tiled_land = np.tile(self.data.reshape(1, -1), n)
		
		prod1 = tiled_data * tiled_land
		sum1 = tiled_data + tiled_land
		
		power = np.sum(np.vstack(np.split(prod1 / sum1, self.N * n, axis=1)), axis=1)
		output = power.reshape(n, self.N)
		
		return output.T
	
	def kernel_single_eval(self, new_data):
		
		"""
		Evaluates the chi-square kernel function at new_data involving double `for` loops.
		This approach is less efficient comparing to self.kernel_matrix.
		Each entry is k(X_i, Y_j), where X_i corresponds to the i-th row of data,
		Y_j corresponds to the j-th row of new_data.

		Parameters
		----------
		new_data : numpy.ndarray
			The array at which the chi-square kernel function is to evaluated.

		Returns
		-------
		numpy.ndarray
			The array of chi-square kernel function evaluations.

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
				s = 0
				for k in range(self.d):
					s += 2. * self.data[i][k] * new_data[j][k] / (self.data[i][k] * new_data[j][k])
				
				output[i][j] = s
				
		return output


if __name__ == '__main__':
	
	data1 = np.random.randn(500 * 3).reshape(500, 3)
	new_data1 = np.random.randn(1000 * 3).reshape(1000, 3)
	kernel = ChiSquare(data1)
	k1 = kernel.kernel_matrix(new_data1)
	k2 = kernel.kernel_single_eval(new_data1)
	print(np.allclose(k1, k2))
	
	data2 = np.random.randn(500)
	new_data2 = np.random.randn(1000)
	kernel = ChiSquare(data2)
	k3 = kernel.kernel_matrix(new_data2)
	k4 = kernel.kernel_single_eval(new_data2)
	print(np.allclose(k3, k4))
