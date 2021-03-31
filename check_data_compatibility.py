import numpy as np


def check_data_compatibility(data, new_data):
	
	"""
	Check whether data and new_data are compatible.
	
	Parameters
	----------
	data, new_data : numpy.ndarray
		The arrays whose compatibilities are to be checked.
		
	"""
	
	data_shape = data.shape
	newdata_shape = new_data.shape
	
	if len(data_shape) != len(newdata_shape):
		
		raise ValueError(f'data and new_data are not compatible. The shape of data is {data_shape}, '
						 f'but that of new_data is {newdata_shape}.')
	
	else:
		
		if len(data_shape) == 2:
			
			if data_shape[1] != newdata_shape[1]:
				
				raise ValueError(f'data and new_data are not compatible. The shape of data is {data_shape}, '
						 	     f'but that of new_data is {newdata_shape}.')
			
			else:
				
				pass


if __name__ == "__main__":

	data = np.random.randn(20 * 2).reshape(20, 2)
	new_data = np.random.randn(100 * 2).reshape(100, 2)
	check_data_compatibility(data, new_data)
