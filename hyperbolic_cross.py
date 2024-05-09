import numpy as np


def hyperbolic_cross(orders, n):
	
	dimensions = len(orders)

	I = np.arange(n+1)
	I = np.reshape(I, (1,-1))

	for k in range(2, dimensions+1):
		J = np.array([]).reshape((I.shape[0]+1, 0))
		for i in range(n+1):
			l = I.shape[1]
			for j in range(l):
				z = I[:,j]
				#z = np.reshape(z, (I.shape[0], 1))

				if (i+1)**orders[k-1]*np.prod(np.power(z, np.array(orders[:k-1]))+1) <= n+1:

					z = np.row_stack((z.reshape((-1,1)), np.array([i]).reshape(1,1)))					
					J = np.hstack((J, z))
		 
		I = J

	return I


