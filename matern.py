from scipy.special import kv
from scipy.special import gamma
from check_data_compatibility import *

class Matern:
	
	"""
	This is a class of evaluating the Matern kernel function.

	Attributes
	----------
	data : numpy.ndarray
		The array at which the Matern kernel function is to evaluated.
	N, d : int, int
		The shape of data array.
	nu : float
		The smoothness parameter in the Matern kernel function; must be strictly greater than d/2.
	bandwidth : float
		The bandwidth parameter in the Matern kernel function; must be strictly positive.

	Methods
	-------
	kernel_matrix(new_data)
		Evaluates the Matern kernel function at new_data.

	kernel_single_eval(new_data)
		Evaluates the Matern kernel function at new_data involving double `for` loops;
		less efficient than self.kernel_matrix.

	"""
	
	def __init__(self, data, nu, bw):
		
		"""
		Parameters
		----------
		data : numpy.ndarray
			The array at which the Matern kernel function is to evaluated.
		nu : float
			The smoothness parameter in the Matern kernel function; must be strictly greater than d/2.
		bw : float
			The bandwidth parameter in the Matern kernel function; must be strictly positive.
		
		"""
		
		if not isinstance(data, np.ndarray):
			data = np.array(data)
		
		if len(data.shape) == 1:
			data = data.reshape(-1, 1)
		
		self.data = data
		self.N, self.d = self.data.shape
		
		if nu <= self.d / 2.:
			raise ValueError(f'The smoothness parameter, nu, must be strictly greater than {self.d/2.}.')
		else:
			self.nu = nu
			
		if bw <= 0.:
			raise ValueError(f'The bandwidth parameter, bw, must be strictly positive.')
		else:
			self.bw = bw
	
	def kernel_matrix(self, new_data):
		
		"""
		Evaluates the Matern kernel function at new_data.
		Each entry is k(X_i, Y_j), where X_i corresponds to the i-th row of data,
		Y_j corresponds to the j-th row of new_data.

		Parameters
		----------
		new_data : numpy.ndarray
			The array at which the Matern kernel function is to evaluated.

		Returns
		-------
		numpy.ndarray
			The array of Matern kernel function evaluations.

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
		
		diff = (tiled_data - tiled_land) ** 2 / (self.bw ** 2)
		power = np.sum(np.vstack(np.split(diff, self.N * n, axis=1)), axis=1)
		power = power.reshape(n, self.N)
		
		dist = np.sqrt(2 * self.nu) * np.sqrt(power)
		output = 2 ** (1 - self.nu) / gamma(self.nu) * dist ** self.nu * kv(self.nu, dist)
		
		return output.T
	
	def kernel_single_eval(self, new_data):
		
		"""
		Evaluates the Matern kernel function at new_data involving double `for` loops.
		This approach is less efficient comparing to self.kernel_matrix.
		Each entry is k(X_i, Y_j), where X_i corresponds to the i-th row of data,
		Y_j corresponds to the j-th row of new_data.

		Parameters
		----------
		new_data : numpy.ndarray
			The array at which the Matern kernel function is to evaluated.

		Returns
		-------
		numpy.ndarray
			The array of Matern kernel function evaluations.

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
				dist = np.sqrt(2 * self.nu) * np.sqrt(np.sum((self.data[i] - new_data[j]) ** 2 / self.bw ** 2))
				output[i][j] = 2. ** (1. - self.nu) / gamma(self.nu) * dist ** self.nu * kv(self.nu, dist)
		
		return output


if __name__ == '__main__':

	data1 = np.random.randn(500 * 5).reshape(500, 5)
	new_data1 = np.random.randn(1000 * 5).reshape(1000, 5)
	kernel = Matern(data1, 3.5, 0.543)
	k1 = kernel.kernel_matrix(new_data1)
	k2 = kernel.kernel_single_eval(new_data1)
	print(np.allclose(k1, k2))
	
	data2 = np.random.randn(500)
	new_data2 = np.random.randn(1000)
	kernel = Matern(data2, 2.5, 1.657)
	k3 = kernel.kernel_matrix(new_data2)
	k4 = kernel.kernel_single_eval(new_data2)
	print(np.allclose(k3, k4))
