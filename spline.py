import numpy as np
from check_data_compatibility import *

class Spline:
	
	"""
	This is a class of evaluating the spline kernel function.

	Attributes
	----------
	data : numpy.ndarray
		The array at which the spline kernel function is to evaluated.
	N, d : int, int
		The shape of data array.

	Methods
	-------
	kernel_matrix(new_data)
		Evaluates the spline kernel function at new_data.

	kernel_single_eval(new_data)
		Evaluates the spline kernel function at new_data involving double `for` loops;
		less efficient than self.kernel_matrix.

	"""
	
	def __init__(self, data):
		
		"""
		Parameters
		----------
		data : numpy.ndarray
			The array at which the spline kernel function is to evaluated.
		
		"""
		
		if not isinstance(data, np.ndarray):
			data = np.array(data)
		
		if len(data.shape) == 1:
			data = data.reshape(-1, 1)
		
		self.data = data
		self.N, self.d = self.data.shape
	
	def kernel_matrix(self, new_data):
		
		"""
		Evaluates the spline kernel function at new_data.
		Each entry is k(X_i, Y_j), where X_i corresponds to the i-th row of data,
		Y_j corresponds to the j-th row of new_data.

		Parameters
		----------
		new_data : numpy.ndarray
			The array at which the spline kernel function is to evaluated.

		Returns
		-------
		numpy.ndarray
			The array of spline kernel function evaluations.

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
		min1 = np.min(np.vstack([tiled_data, tiled_land]), axis=0).reshape(1, -1)
		
		output = 1. + prod1 + prod1 * min1 - sum1 / 2. * min1 ** 2 + min1 ** 3 / 3.
		output = np.prod(np.vstack(np.split(output.flatten(), self.N * n)), axis=1)
		output = output.reshape(n, self.N)
		
		return output.T
	
	def kernel_single_eval(self, new_data):
		
		"""
		Evaluates the spline kernel function at new_data involving double `for` loops.
		This approach is less efficient comparing to self.kernel_matrix.
		Each entry is k(X_i, Y_j), where X_i corresponds to the i-th row of data,
		Y_j corresponds to the j-th row of new_data.

		Parameters
		----------
		new_data : numpy.ndarray
			The array at which the spline kernel function is to evaluated.

		Returns
		-------
		numpy.ndarray
			The array of spline kernel function evaluations.

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
				s = 1.
				dd = self.data[i]
				nd = new_data[j]
				for k in range(self.d):
					mind = np.min([dd[k], nd[k]])
					s *= 1. + dd[k] * nd[k] + dd[k] * nd[k] * mind - (dd[k] + nd[k]) / 2 * mind ** 2 + mind ** 3 / 3
				output[i][j] = s
		
		return output


if __name__ == '__main__':
	
	data1 = np.random.randn(500 * 3).reshape(500, 3)
	new_data1 = np.random.randn(1000 * 3).reshape(1000, 3)
	kernel = Spline(data1)
	k1 = kernel.kernel_matrix(new_data1)
	k2 = kernel.kernel_single_eval(new_data1)
	print(np.allclose(k1, k2))
	
	data2 = np.random.randn(500)
	new_data2 = np.random.randn(1000)
	kernel = Spline(data2)
	k3 = kernel.kernel_matrix(new_data2)
	k4 = kernel.kernel_single_eval(new_data2)
	print(np.allclose(k3, k4))
