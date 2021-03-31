import numpy as np
from check_data_compatibility import *

class Power:
	
	"""
	This is a class of evaluating the power kernel function.

	Attributes
	----------
	data : numpy.ndarray
		The array at which the power kernel function is to evaluated.
	N, d : int, int
		The shape of data array.
	degree : int
		The degree of the power kernel function; must be a non-negative integer.

	Methods
	-------
	kernel_matrix(new_data)
		Evaluates the power kernel function at new_data.

	kernel_single_eval(new_data)
		Evaluates the power kernel function at new_data involving double `for` loops;
		less efficient than self.kernel_matrix.

	"""
	
	def __init__(self, data, degree):
		
		"""
		Parameters
		----------
		data : numpy.ndarray
			The array at which the degree kernel function is to evaluated.
		degree : int
			The degree of the power kernel function; must be a non-negative integer.
		
		"""
		
		if not isinstance(data, np.ndarray):
			data = np.array(data)
		
		if len(data.shape) == 1:
			data = data.reshape(-1, 1)
		
		self.data = data
		self.N, self.d = self.data.shape
		if not isinstance(degree, int):
			raise TypeError(f'The degree must be an integer.')
		
		if degree < 0:
			raise ValueError(f'The degree must be non-negative. ')
		else:
			self.degree = degree
	
	def kernel_matrix(self, new_data):
		
		"""
		Evaluates the power kernel function at new_data.
		Each entry is k(X_i, Y_j), where X_i corresponds to the i-th row of data,
		Y_j corresponds to the j-th row of new_data.

		Parameters
		----------
		new_data : numpy.ndarray
			The array at which the power kernel function is to evaluated.

		Returns
		-------
		numpy.ndarray
			The array of power kernel function evaluations.

		"""
		
		if not isinstance(new_data, np.ndarray):
			new_data = np.array(new_data)
		
		if len(new_data.shape) == 1:
			new_data = new_data.reshape(-1, 1)
		
		n, d1 = new_data.shape
		
		check_data_compatibility(self.data, new_data)
		
		tiled_data = np.tile(new_data, self.N).reshape(1, -1)
		tiled_land = np.tile(self.data.reshape(1, -1), n)
		
		diff = (tiled_data - tiled_land) ** 2
		power = np.sum(np.vstack(np.split(diff, self.N * n, axis=1)), axis=1)
		power = power.reshape(n, self.N)
		
		power_part = -np.sqrt(power) ** self.degree
		
		return power_part.T
	
	def kernel_single_eval(self, new_data):
		
		"""
		Evaluates the power kernel function at new_data involving double `for` loops.
		This approach is less efficient comparing to self.kernel_matrix.
		Each entry is k(X_i, Y_j), where X_i corresponds to the i-th row of data,
		Y_j corresponds to the j-th row of new_data.

		Parameters
		----------
		new_data : numpy.ndarray
			The array at which the power kernel function is to evaluated.

		Returns
		-------
		numpy.ndarray
			The array of power kernel function evaluations.

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
				output[i][j] = - np.sqrt(np.sum((self.data[i] - new_data[j]) ** 2)) ** self.degree
		
		return output


if __name__ == '__main__':

	data1 = np.random.randn(500 * 3).reshape(500, 3)
	new_data1 = np.random.randn(1000 * 3).reshape(1000, 3)
	kernel = Power(data1, 3)
	k1 = kernel.kernel_matrix(new_data1)
	k2 = kernel.kernel_single_eval(new_data1)
	print(np.allclose(k1, k2))
	
	data2 = np.random.randn(500)
	new_data2 = np.random.randn(1000)
	kernel = Power(data2, 4)
	k3 = kernel.kernel_matrix(new_data2)
	k4 = kernel.kernel_single_eval(new_data2)
	print(np.allclose(k3, k4))
