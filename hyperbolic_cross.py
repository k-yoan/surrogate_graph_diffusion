
def hyperbolic_cross(orders):

	n = orders[0] # ???
	dimensions = len(orders)

	I = np.arange(n+1)
	I = np.reshape(I, (1,-1))

	for k in range(2, dimensions+1):
		J = np.zeros((I.shape[0]+1, 1))
		for i in range(n+1):
			l = I.shape[1]
			for j in range(l):
				z = I[:,j]
				#z = np.reshape(z, (I.shape[0], 1))

				if (i+1)*np.prod(z+1) <= n+1:
					print(z.shape)
					z = np.row_stack((z, i))
					print(z.shape)
					#J = np.row_stack((J, 0))
					print(J.shape)
					J = np.hstack((J, z))
		 
		I = J

	return I


