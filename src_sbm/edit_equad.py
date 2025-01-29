''' There are some incompatibilities between the equadratures library and the desired setup for our numerical experiments.
This file is used to edit lines of code from the necessary files in the equadratures library to allow us to run the experiments without issues. '''

from get_lib_path import *


# First, the equadratures library does not have a hyperbolic cross object. We must insert a function to create a hyperbolic cross object.

path_to_library = get_library_path('equadratures')

new_code = '''
def hyperbolic_cross(orders):
    dimensions = len(orders)
    n = orders[0]
    I = np.arange(n+1)
    I = np.reshape(I, (1,-1))
    for k in range(2, dimensions+1):
        J = np.array([]).reshape((I.shape[0]+1, 0))
        for i in range(n+1):
            l = I.shape[1]
            for j in range(l):
                z = I[:,j]
                #z = np.reshape(z, (I.shape[0], 1))
                if (i+1)*np.prod(z+1) <= n+1:
                    z = np.row_stack((z.reshape((-1,1)), np.array([i]).reshape(1,1)))
                    J = np.hstack((J, z))
        I = J
    return I.T
'''

# The new hyperbolic cross object is inserted in the basis.py file

with open(path_to_library+"/basis.py", "a") as file:
  file.write(new_code)


# Then, a few lines must also be inserted in other files in order to still be able to use
# commands related to basis objects for the newly created hyperbolic cross object.
# The insertion is done from the bottom to the top, to avoid messing up w/ the line numbers.

new_line_1 = '''        elif name == "hyperbolic-cross":
            basis = hyperbolic_cross(self.orders)
'''

# Open the file in read mode to read its contents
with open(path_to_library+"/basis.py", 'r') as file:
    lines = file.readlines()

lines.insert(179, new_line_1)

# Open the file in write mode to overwrite its contents
with open(path_to_library+"/basis.py", 'w') as file:
    file.writelines(lines)

new_line_2 = '''        elif name.lower() == "hyperbolic-cross":
            basis = hyperbolic_cross(self.orders)
'''

# Open the file in read mode to read its contents
with open(path_to_library+"/basis.py", 'r') as file:
    lines = file.readlines()

lines.insert(93, new_line_2)

# Open the file in write mode to overwrite its contents
with open(path_to_library+"/basis.py", 'w') as file:
    file.writelines(lines)


with open(path_to_library+"/optimisation.py", "r") as file:
    fix_1 = file.read()

# Lines to modify in equadratures/optimisation.py

fix_1 = fix_1.replace('objective = lambda x: k*np.asscalar(f(x))', 'objective = lambda x: k*f(x).item()')
fix_1 = fix_1.replace('constraint = lambda x: np.asscalar(g(x))', 'constraint = lambda x: g(x).item()')
fix_1 = fix_1.replace('self.f_old = np.asscalar(self.f[ind_min])', 'self.f_old = self.f[ind_min].item()')
fix_1 = fix_1.replace('return np.asscalar(f)', 'return f.item()')
fix_1 = fix_1.replace('np.asscalar(my_poly.get_polyfit(x))', 'my_poly.get_polyfit(x).item()')
fix_1 = fix_1.replace('np.asscalar(my_poly.get_polyfit(np.dot(x,self.U)))', 'my_poly.get_polyfit(np.dot(x,self.U)).item()')
fix_1 = fix_1.replace('del_m = np.asscalar(my_poly.get_polyfit(self.s_old)) - m_new', 'del_m = np.ndarray.item(my_poly.get_polyfit(self.s_old)) - m_new')
fix_1 = fix_1.replace('del_m = np.asscalar(my_poly.get_polyfit(np.dot(self.s_old,self.U))) - m_new', 'del_m = np.ndarray.item(my_poly.get_polyfit(np.dot(self.s_old,self.U))) - m_new')

with open(path_to_library+"/optimisation.py", "w") as file:
    file.write(fix_1)


with open(path_to_library+"/sampling_methods/induced.py", "r") as file:
    fix_2 = file.read()

# Lines to modify in equadratures/sampling_methods/induced.py

fix_2 = fix_2.replace('F = np.asscalar(F)', 'F = F.item()')

with open(path_to_library+"/sampling_methods/induced.py", "w") as file:
    file.write(fix_2)


with open(path_to_library+"/solver.py", "r") as file:
    fix_3 = file.read()

# Lines to modify in equadratures/solver.py

fix_3 = fix_3.replace('fe = 0.5*(np.asscalar(np.dot(r.T,r)) - epsilon**2)', 'fe = 0.5*(np.dot(r.T,r).item() - epsilon**2)')
fix_3 = fix_3.replace('cqe = np.asscalar(np.dot(r.T,r)) - epsilon**2', 'cqe = np.dot(r.T,r).item() - epsilon**2')

with open(path_to_library+"/solver.py", "w") as file:
    file.write(fix_3)

with open(path_to_library+"/subspaces.py", "r") as file:
    fix_4 = file.read()

# Lines to modify in equadratures/subspaces.py

fix_4 = fix_4.replace('dV[:,l,:,j] = np.asscalar(vectord[l])*(X.T*current[:,j])', 'dV[:,l,:,j] = vectord[l].item()*(X.T*current[:,j])')

with open(path_to_library+"/subspaces.py", "w") as file:
    file.write(fix_4)




