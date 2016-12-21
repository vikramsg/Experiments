# Need to import the plotting package:
import matplotlib.pyplot as plt
import numpy as np

# Read the file. 
f2 = open('out.dat', 'r')
# read the whole file into a single variable, which is a list of every row of the file.
lines1 = f2.readlines()
f2.close()

# initialize some variable to be lists:
x1 = []
y1 = []
y2 = []

# scan the rows of the file stored in lines, and put the values into some variables:
for line in lines1:
    p = line.split()
    x1.append(float(p[0]))
    y1.append(float(p[1]))
    y2.append(float(p[2]))
  
xv1 = np.array(x1)
yv1 = np.array(y1)
yv2 = np.array(y2)

exv1  = xv1 
coeff = (2*np.pi/(1.0))
#eyv1  = -coeff*coeff*np.sin(coeff*exv1)
c = 100; alpha = 0.02; L = 1; t = 10
eyv1  =  c*np.exp((-alpha*np.pi**2*t)/(L**2))*np.sin(np.pi*exv1/L)


# now, plot the data:
plt.plot(xv1, yv1, color = "blue", linewidth =  2.5, \
         linestyle = "-" )
plt.plot(xv1, yv2, color = "blue", linewidth =  2.5, \
         linestyle = "dotted")
#plt.plot(exv1, eyv1, color = "red", linewidth =  2.5, \
#         linestyle = "dashed")

print(np.max(yv1))

plt.ylim([-5, 5])

#plt.title('n = '+str(n)+', cfl = '+str(cfl))
#plt.xlabel('x')
#plt.legend(loc='upper left', fontsize = 10)
#plt.legend(loc='upper center', bbox_to_anchor = (0.95, 1.12), fontsize = 10)

plt.show()
#plt.savefig('figure_n_'+str(n)+'_cfl_'+str(cfl)+'.png')

