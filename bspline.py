import numpy as np
from math import factorial
from scipy.special import comb
from check_data_compatibility import *

class BSpline:
	
	"""
	This is a class of evaluating the B-spline kernel function.

	Attributes
	----------
	data : numpy.ndarray
		The array at which the B-spline kernel function is to evaluated.
	N, d : int, int
		The shape of data array.
	order : int
		The order of the B-spline kernel function; must be a non-negative integer.

	Methods
	-------
	kernel_matrix(new_data)
		Evaluates the B-spline kernel function at new_data.

	kernel_single_eval(new_data)
		Evaluates the B-spline kernel function at new_data involving double `for` loops;
		less efficient than self.kernel_matrix.

	"""
	
	def __init__(self, data, order):
		
		"""
		Parameters
		----------
		data : numpy.ndarray
			The array at which the B-spline kernel function is to evaluated;
			all entries must be between -1 and 1.
		order : int
			The order of the B-spline kernel function; must be a non-negative integer.
		
		"""
		
		if np.max(data > 1.) or np.min(data < -1.):
			raise ValueError(f'The data must be between -1 and 1.')
		
		if not isinstance(data, np.ndarray):
			data = np.array(data)
		
		if len(data.shape) == 1:
			data = data.reshape(-1, 1)
		
		self.data = data
		self.N, self.d = self.data.shape
		
		if not isinstance(order, int):
			raise ValueError(f'The order must be an integer.')
		elif order < 0:
			raise ValueError(f'The order must be non-negative.')
		else:
			self.order = order
	
	def kernel_matrix(self, new_data):
		
		"""
		Evaluates the B-spline kernel function at new_data.
		Each entry is k(X_i, Y_j), where X_i corresponds to the i-th row of data,
		Y_j corresponds to the j-th row of new_data.

		Parameters
		----------
		new_data : numpy.ndarray
			The array at which the B-spline kernel function is to evaluated.

		Returns
		-------
		numpy.ndarray
			The array of B-spline kernel function evaluations.

		"""
		
		if np.max(new_data > 1.) or np.min(new_data < -1.):
			raise ValueError(f'The new_data must be between -1 and 1.')
		
		if not isinstance(new_data, np.ndarray):
			new_data = np.array(new_data)
		
		if len(new_data.shape) == 1:
			new_data = new_data.reshape(-1, 1)
		
		n, d1 = new_data.shape
		
		check_data_compatibility(self.data, new_data)
		
		tiled_data = np.tile(new_data, self.N).reshape(1, -1)
		tiled_land = np.tile(self.data.reshape(1, -1), n)
		diff = (tiled_data - tiled_land).flatten()
		
		order = self.order
		result = 0.
		for i in range(order + 2):
			result += comb(order + 1, i) * (-1) ** i * ((diff + (order + 1) / 2. - i) * ((diff + (order + 1) / 2. - i) >= 0)) ** order / factorial(order)
		
		result = np.prod(np.vstack(np.split(result, self.N * n)), axis=1)
		result = result.reshape(n, self.N)
		
		return result.T
	
	def kernel_single_eval(self, new_data):
		
		"""
		Evaluates the B-spline kernel function at new_data involving double `for` loops.
		This approach is less efficient comparing to self.kernel_matrix.
		Each entry is k(X_i, Y_j), where X_i corresponds to the i-th row of data,
		Y_j corresponds to the j-th row of new_data.

		Parameters
		----------
		new_data : numpy.ndarray
			The array at which the B-spline kernel function is to evaluated.

		Returns
		-------
		numpy.ndarray
			The array of B-spline kernel function evaluations.

		"""
		
		if np.max(new_data > 1.) or np.min(new_data < -1.):
			raise ValueError(f'The new_data must be between -1 and 1.')
		
		if not isinstance(new_data, np.ndarray):
			new_data = np.array(new_data)
		
		if len(new_data.shape) == 1:
			new_data = new_data.reshape(-1, 1)
			
		n, d1 = new_data.shape
		
		check_data_compatibility(self.data, new_data)
		
		output = np.zeros((self.N, n), dtype=np.float64)
		order = self.order
		
		for i in range(self.N):
			for j in range(n):
				diff = self.data[i] - new_data[j]
				s = 0.
				for k in range(order + 2):
					s += (comb(order + 1, k) * (-1) ** k * ((diff + (order + 1) / 2. - k) * (diff + (order + 1) / 2. - k >= 0)) ** order /
						  factorial(order))
				
				output[i][j] = np.prod(s)
		
		return output


if __name__ == '__main__':
	
	data1 = np.random.rand(500 * 3).reshape(500, 3)
	new_data1 = np.random.rand(1000 * 3).reshape(1000, 3)
	kernel = BSpline(data1, 2)
	k1 = kernel.kernel_matrix(new_data1)
	k2 = kernel.kernel_single_eval(new_data1)
	print(np.allclose(k1, k2))
	
	data2 = np.random.rand(500)
	new_data2 = np.random.rand(1000)
	kernel = BSpline(data2, 2)
	k3 = kernel.kernel_matrix(new_data2)
	k4 = kernel.kernel_single_eval(new_data2)
	print(np.allclose(k3, k4))
