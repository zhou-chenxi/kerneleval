import numpy as np
from check_data_compatibility import *

class Spherical:
	
	"""
	This is a class of evaluating the spherical kernel function.

	Attributes
	----------
	data : numpy.ndarray
		The array at which the spherical kernel function is to evaluated.
	N, d : int, int
		The shape of data array.
	bw : float
		The bandwidth parameter in the spherical kernel function; must be strictly positive.

	Methods
	-------
	kernel_matrix(new_data)
		Evaluates the spherical kernel function at new_data.

	kernel_single_eval(new_data)
		Evaluates the spherical kernel function at new_data involving double `for` loops;
		less efficient than self.kernel_matrix.

	"""
	
	def __init__(self, data, bw):
		
		"""
		Parameters
		----------
		data : numpy.ndarray
			The array at which the spherical kernel function is to evaluated.
		bw : float
			The bandwidth parameter in the spherical kernel function; must be strictly positive.
		
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
	
	def kernel_matrix(self, new_data):
		
		"""
		Evaluates the spherical kernel function at new_data.
		Each entry is k(X_i, Y_j), where X_i corresponds to the i-th row of data,
		Y_j corresponds to the j-th row of new_data.

		Parameters
		----------
		new_data : numpy.ndarray
			The array at which the spherical kernel function is to evaluated.

		Returns
		-------
		numpy.ndarray
			The array of spherical kernel function evaluations.

		"""
		
		if not isinstance(new_data, np.ndarray):
			new_data = np.array(new_data)
		
		if len(new_data.shape) == 1:
			new_data = new_data.reshape(-1, 1)
		
		n, d1 = new_data.shape
		
		check_data_compatibility(self.data, new_data)
		
		tiled_data = np.tile(new_data, self.N).reshape(1, -1)
		tiled_land = np.tile(self.data.reshape(1, -1), n)
		
		diff = (tiled_data - tiled_land) ** 2 / self.bw ** 2
		power = np.sum(np.vstack(np.split(diff, self.N * n, axis=1)), axis=1)
		power = power.reshape(n, self.N)
		
		sqrt_part = np.sqrt(power) * (power <= 1.0)
		sph_part = 1. - (3. / 2.) * sqrt_part + (1. / 2.) * (sqrt_part) ** 3
		sph_part = sph_part * (power <= 1.0)
		
		return sph_part.T
	
	def kernel_single_eval(self, new_data):
		
		"""
		Evaluates the spherical kernel function at new_data involving double `for` loops.
		This approach is less efficient comparing to self.kernel_matrix.
		Each entry is k(X_i, Y_j), where X_i corresponds to the i-th row of data,
		Y_j corresponds to the j-th row of new_data.

		Parameters
		----------
		new_data : numpy.ndarray
			The array at which the spherical kernel function is to evaluated.

		Returns
		-------
		numpy.ndarray
			The array of spherical kernel function evaluations.

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
				sqrt_norm = np.sqrt(np.sum((self.data[i] - new_data[j]) ** 2) / self.bw ** 2)
				if sqrt_norm > 1.:
					output[i][j] = 0.
				else:
					output[i][j] = 1. - (3. / 2.) * sqrt_norm + 1. / 2. * sqrt_norm ** 3
		
		return output


if __name__ == '__main__':
	
	data1 = np.random.randn(500 * 3).reshape(500, 3)
	new_data1 = np.random.randn(1000 * 3).reshape(1000, 3)
	kernel = Spherical(data1, 0.43)
	k1 = kernel.kernel_matrix(new_data1)
	k2 = kernel.kernel_single_eval(new_data1)
	print(np.allclose(k1, k2))
	
	data2 = np.random.randn(500)
	new_data2 = np.random.randn(1000)
	kernel = Spherical(data2, 2.3)
	k3 = kernel.kernel_matrix(new_data2)
	k4 = kernel.kernel_single_eval(new_data2)
	print(np.allclose(k3, k4))
